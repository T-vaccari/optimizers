import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.main import adam




torch.manual_seed(42)

ours_param = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
torch_param = ours_param.detach().clone().requires_grad_(True)

ours = adam([ours_param], lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)
reference = torch.optim.Adam(
    [torch_param], lr=1e-3, betas=(0.9, 0.999), eps=1e-8
)

for step in range(10):
    grad = torch.randn_like(ours_param)
    ours_param.grad = grad.clone()
    torch_param.grad = grad.clone()

    ours.step()
    reference.step()

    torch.testing.assert_close(ours_param, torch_param, rtol=1e-12, atol=1e-12)
    ours.zero_grad()
    reference.zero_grad()

print("Test passed!")
