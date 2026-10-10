import os

# lightning reloads the (own) best checkpoint for testing, which torch>=2.6 refuses by default
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import tvsnr_mol  # noqa: F401,E402
from schnetpack.cli import train  # noqa: E402

if __name__ == "__main__":
    train()
