import torch
import numpy as np
from tv_snr.noise_schedules import CosineSchedule
from tv_snr.snr_schedules import *
from tv_snr.sdes import *
from tv_snr.sampler import *
from tv_snr.constants import image_key
import matplotlib.pyplot as plt

from tv_snr.time_schedules import KVeSchedule
from tv_snr.scale_schedule import *
from tv_snr.adaptive_scale_sampler import *


# Rephrasing EDM ODE with linear t schedule
def tv_snr_sampler_example(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    compute_sigma_dot="fd"
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    A = sigma_max ** (1 / rho)
    B = sigma_min ** (1 / rho) - sigma_max ** (1 / rho)

    def time(i):
        # linear time schedule
        return i / (num_steps - 1)
    
    def sigma(i):
        if i < num_steps:
            return (A + time(i) * B) ** rho
        else:
            return 0
        
    def sigma_dot(i):
        def _sigma_f(t):
            return (A + t * B) ** rho

        if compute_sigma_dot == "fd":
            # dSigma / dt via finite differences
            return (sigma(i + 1) - sigma(i)) / (time(i + 1) - time(i))
        elif compute_sigma_dot == "at":
            # analytic gradient
            return B * rho * (A + time(i) * B) ** (rho - 1)

            # this is equivalent to the above
            # t_cur = torch.tensor(time(i)).requires_grad_(True)
            # grad = torch.autograd.grad(_sigma_f(t_cur).sum(), t_cur)[0]
            # return grad
        else:
            assert compute_sigma_dot == "athalf"
            # analytic gradient at half step
            t_half = time(i) + (time(i + 1) - time(i)) / 2
            return B * rho * (A + t_half * B) ** (rho - 1)

    # Main sampling loop.
    x_next = latents.to(torch.float64) * sigma(0)
    for i in range(num_steps):
        x_hat = x_next

        # note that this might not match the sigma^(-1) schedule exactly
        # since sigma is rounded by the network
        dt = time(i + 1) - time(i)
        sigma_hat = net.round_sigma(sigma(i)).to(latents.device)
        sigma_next = net.round_sigma(sigma(i + 1)).to(latents.device)
        sigma_deriv = sigma_dot(i)

        # Euler step.
        denoised = net(x_hat, sigma_hat, class_labels).to(torch.float64)
        d_cur = sigma_deriv * (x_hat - denoised) / sigma_hat
        x_next = x_hat + dt * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            sigma_deriv_next = sigma_dot(i)

            denoised = net(x_next, sigma_next, class_labels).to(torch.float64)
            d_prime = sigma_deriv_next * (x_next - denoised) / sigma_next
            x_next = x_hat + dt * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


# Using the tv_snr framework
def tv_snr_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    tau=0, disc_type="forward", solver="heun", linear_time=False, snr_schedule="linear"
):
    adaptive = False

    if snr_schedule == "linear":
        assert not linear_time, "Linear time schedule is not supported with linear SNR schedule."
        snr_sch = LinearToSNRSchedule()
    elif snr_schedule == "cosine2.5":
        assert linear_time, "Cosine SNR schedule requires linear time schedule. (for now)"
        noise_sch = CosineSchedule(v=2.5, s=0.008, T=num_steps, discretize=True)
        snr_sch = NoiseToSNRSchedule(noise_sch,t_min=0.03,t_max=.9968)
    elif snr_schedule == "cosine1.5":
        assert linear_time, "Cosine SNR schedule requires linear time schedule. (for nowwhy)"
        noise_sch = CosineSchedule(v=1.5, s=0.008, T=num_steps, discretize=True)
        snr_sch = NoiseToSNRSchedule(noise_sch,t_min=0.03,t_max=.9968)
    elif snr_schedule == "sig":
        assert linear_time, "Sigmoid SNR schedule requires linear time schedule. (for now)"
        snr_sch = InverseSigmoid(slope=5., shift=-1., t_min=5e-2, t_max=0.83)
    elif snr_schedule == "sig2":
        assert linear_time, "Sigmoid SNR schedule requires linear time schedule. (for now)"
        snr_sch = InverseSigmoid(slope=5., shift=1., t_min=0.0923, t_max=0.875753)
    elif snr_schedule == "sig3":
        # assert linear_time, "Sigmoid SNR schedule requires linear time schedule. (for now)"
        snr_sch = InverseSigmoid(slope=3., shift=2., t_min=0.03, t_max=0.973)
    elif snr_schedule == "straight":
        # this is FM with tau(t)=1
        # assert linear_time, "StraightEstimatorToSNRSchedule requires linear time schedule. (for now)"
        assert disc_type=="forward", "StraightEstimatorToSNRSchedule works only with forward discretization."
        assert tau==1.0, "StraightEstimatorToSNRSchedule works only with tau=1.0."
        #snr_sch = StraightEstimatorToSNRSchedule(t_min=(1/501),t_max=(80/81))
        snr_sch = InverseSigmoid(slope=2., shift=0., t_min=(1/501), t_max=(80/81))
    elif snr_schedule == "fm_adaptive":
        # assert linear_time, "FM with adaptive tau(t) requires linear time schedule."
        # we can use Karras schedule with max_sigma = 1
        assert disc_type=="forward", "FM with adaptive tau(t) works only with forward discretization."
        assert tau==1.0, "FM with adaptive tau(t) works only with tau=1.0 (tau is adaptive though)."

        adaptive = True

        # this is FM with adaptive tau(t)
        snr_sch = InverseSigmoid(slope=2., shift=0., t_min=(1/501), t_max=(80/81))
        scale_sch = FMScale(snr_sch)
        sde = Scale_SNR_SDE(
            snr_sch=snr_sch,
            scale_sch=scale_sch,
            invariant=False,
            disc_type="forward",
            log_deriv=True
        )
        sampler_cls = SNRHeunAdaptiveScale if solver == "heun" else SNREulerAdaptiveScale
    elif snr_schedule == "sig3_adaptive":
        assert linear_time, "sig3 with adaptive tau(t) requires linear time schedule."
        assert disc_type=="forward", "sig3 with adaptive tau(t) works only with forward discretization."
        assert tau==1.0, "sig3 with adaptive tau(t) works only with tau=1.0 (tau is adaptive though)."

        adaptive = True

        # this is FM with adaptive tau(t)
        snr_sch = InverseSigmoid(slope=3., shift=2., t_min=0.03, t_max=0.973)
        scale_sch = FMScale(snr_sch)
        sde = Scale_SNR_SDE(
            snr_sch=snr_sch,
            scale_sch=scale_sch,
            invariant=False,
            disc_type="forward",
            log_deriv=True
        )
        sampler_cls = SNRHeunAdaptiveScale if solver == "heun" else SNREulerAdaptiveScale
    elif snr_schedule == "ve":
        assert linear_time, "ve schedule requires linear time schedule. (for now)"
        assert disc_type=="forward", "ve schedule works only with forward discretization."
        snr_sch = VeToSNRSchedule(t_min=0, t_max=1.0, sigma_max=80.0)
    else:
        assert linear_time, "KVE time schedule requires linear time schedule."
        assert snr_schedule == "kve"
        snr_sch = KveToSNRSchedule(t_min=0, t_max=1.0, sigma_max=80.0)

    if not adaptive:
        sde = SNR_SDE(
            snr_sch=snr_sch,
            tau=tau,
            invariant=False,
            disc_type=disc_type,
            log_deriv=True
        )
        sampler_cls = SNRHeun if solver == "heun" else SNREuler

    rsde = sde.reverse(stochastic=S_churn > 0)

    if snr_schedule == "linear":
        kve_schedule = KVeSchedule(sigma_max=80.0, discretize=True, T=num_steps)
    else:
        # snr schedule includes sigma scaling already
        kve_schedule = KVeSchedule(sigma_max=1, sigma_min=1/num_steps, discretize=True, T=num_steps, rho=2)

    sampler = sampler_cls(
        T = num_steps,
        time_schedule=None if linear_time else kve_schedule,
        reverse_process = rsde,
        denoiser = net,
        out_var_scaler=1.0,
        scale_input=False,
        snr_key="gamma",
        std_key = "sigma",
        noise_pred_key = "eps_pred",   
        max_stoch_std=S_max,
        min_stoch_std=S_min,
        clip_stoch_std=False,
        selected_stoch=S_churn > 0,
        save_progress=False,
        conditional=False,
    )

    scale_latents = torch.pow(torch.tensor(sigma_max).to("cuda"), 1-tau)
    return sampler.denoise({image_key: latents * scale_latents}, progress_bar=False)[0]["image"]
