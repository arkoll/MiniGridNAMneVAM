"""Сколько параметров и токенов у вариантов, между которыми выбираем."""

import sys

sys.path.insert(0, "/home/user8_2/AIRI_WAM/nano-world-model/src")

import torch  # noqa: E402
from models.nanowm import NanoWM_models  # noqa: E402

VARIANTS = [
    ("NanoWM-S/2", 32, "кадр 256, латент 32"),
    ("NanoWM-S/4", 32, "кадр 256, латент 32"),
    ("NanoWM-S/2", 16, "кадр 128, латент 16"),
    ("NanoWM-B/2", 32, "кадр 256, латент 32"),
]

for arch, latent_size, note in VARIANTS:
    m = NanoWM_models[arch](
        input_size=latent_size,
        in_channels=4,
        num_frames=4,
        use_action=True,
        action_dim=6,
    )
    n = sum(p.numel() for p in m.parameters())
    patch = int(arch.split("/")[1])
    tok_per_frame = (latent_size // patch) ** 2
    print(
        f"{arch:>12} | {note:>22} | параметров {n / 1e6:6.1f} млн | "
        f"токенов на кадр {tok_per_frame:4d} | на окно из 4 кадров {tok_per_frame * 4:5d}"
    )
    del m
print("MODEL_VARIANTS_STATUS=OK")
