import numpy as np
import torch

from tv_snr.constants import image_key, class_labels_key
from tv_snr.samplers import SNREuler, SNRHeun, SNRRK, DPMSolver
from tv_snr.scale_schedules import ConstScale, FMScale, VeScale
from tv_snr.snr import Scale_SNR_SDE
from tv_snr.snr_schedules import (
    InverseSigmoid,
    KveToSNRSchedule,
    LinearToSNRSchedule,
    VeToSNRSchedule,
)
from tv_snr.time_schedules import KVeSchedule


def get_nfe(solver, num_steps):
    if solver == "heun":
        return 2 * num_steps - 1
    # for other solvers nfe is equal to num_steps already
    return num_steps


def eta_scaled_slope(nfe):
    # 2 * eta = 2 + max(0, log2(nfe + 1) - 3), kappa = 0
    return 2.0 + max(0.0, float(np.log2(nfe + 1)) - 3.0)


def issnr_schedule(slope, shift, sigma_min, sigma_max):
    # t_min and t_max such that the SNR spans [sigma_max^-2, sigma_min^-2]
    a = (torch.exp((torch.log(torch.tensor(float(sigma_min)) ** -2) - shift) / slope) + 1) ** (-1)
    b = (torch.exp((torch.log(torch.tensor(float(sigma_max)) ** -2) - shift) / slope) + 1) ** (-1)
    return InverseSigmoid(slope=slope, shift=shift, t_min=a.item(), t_max=b.item())


# Using the TV/SNR framework
def tv_snr_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80.0, rho=7,
    disc_type="forward", solver="heun", snr_schedule="issnr", scale_schedule="constant",
    slope=3.0, shift=2.0, eta_scaling=False,
):
    if solver == "rk45":
        # get atol and rtol from num_steps
        # then set to a constant value
        # directly set num_steps to the (approximate) number of NFE
        # this wont have an influence

        atol = 1e-9
        if num_steps == 3:
            rtol = 1e-2
            num_steps = 63
        elif num_steps == 2:
            rtol = 1e-1
            num_steps = 31
        elif num_steps == 1:
            rtol = 0.3
            num_steps = 31
        elif num_steps == 0:
            rtol = 1.0
            atol = 1.0
            num_steps = 15
        else:
            raise ValueError("Invalid number of steps for RK45 solver.")

    if snr_schedule == "linear":
        # EDM: sigma(t) = t on the Karras time grid
        if solver == "rk45" or solver == "dpm":
            # these solvers don't support time schedules
            snr_sch = LinearToSNRSchedule(sigma_min=sigma_min, sigma_max=sigma_max)
        else:
            snr_sch = LinearToSNRSchedule()
    elif snr_schedule == "kve":
        snr_sch = KveToSNRSchedule(t_min=0, t_max=1.0, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
    elif snr_schedule == "ve":
        snr_sch = VeToSNRSchedule(t_min=0, t_max=1.0, sigma_min=sigma_min, sigma_max=sigma_max)
    elif snr_schedule == "issnr":
        if eta_scaling:
            slope, shift = eta_scaled_slope(get_nfe(solver, num_steps)), 0.0
        snr_sch = issnr_schedule(slope, shift, sigma_min, sigma_max)
    else:
        raise ValueError(f"Unknown SNR schedule {snr_schedule}")

    if scale_schedule == "ve":
        scale_sch = VeScale(snr_sch)
    elif scale_schedule == "constant":
        scale_sch = ConstScale(snr_sch)
    elif scale_schedule == "fm":
        scale_sch = FMScale(snr_sch)
    else:
        raise ValueError(f"Unknown scale schedule {scale_schedule}")

    sde = Scale_SNR_SDE(
        snr_sch=snr_sch,
        scale_sch=scale_sch,
        invariant=False,
        disc_type=disc_type,
        log_deriv=True
    )

    if solver == "heun":
        sampler_cls = SNRHeun
    elif solver == "euler":
        sampler_cls = SNREuler
    elif solver == "rk45":
        sampler_cls = SNRRK
    elif solver == "dpm":
        sampler_cls = DPMSolver
    else:
        raise ValueError(f"Unknown solver {solver}")

    # probability flow ODE
    rsde = sde.reverse(stochastic=False)

    if solver in ["rk45", "dpm"]:
        time_schedule = None
    elif snr_schedule == "linear":
        time_schedule = KVeSchedule(sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, discretize=True, T=num_steps)
    else:
        # uniform time grid
        time_schedule = None

    sampler = sampler_cls(
        T = num_steps,
        time_schedule=time_schedule,
        reverse_process = rsde,
        denoiser = net,
        out_var_scaler=1.0,
        scale_input=False,
        snr_key="gamma",
        std_key = "sigma",
        noise_pred_key = "eps_pred",
        save_progress=False,
        conditional=False,
    )

    # the prior of the VE scale has std sigma_max, otherwise the TV is 1
    scale_latents = sigma_max if scale_schedule == "ve" else 1.0
    run_args = {image_key: latents * scale_latents, class_labels_key: class_labels}
    if class_labels is None:
        del run_args[class_labels_key]

    if solver == "rk45":
        x_0, nfe, _ = sampler.denoise(run_args, progress_bar=False, rtol=rtol, atol=atol)
        return x_0[image_key], nfe
    elif solver == "dpm":
        x_0 = sampler.denoise(run_args, order=3)[0][image_key]
        return x_0, num_steps
    else:
        return sampler.denoise(run_args, progress_bar=False)[0][image_key], get_nfe(solver, num_steps)
