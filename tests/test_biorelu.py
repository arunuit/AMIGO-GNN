import torch
from src.model import BioReLU

def test_biorelu():
    m=BioReLU(0.1)
    x=torch.tensor([-1.0,0.0,1.0])
    y=m(x)
    assert y.shape == x.shape
    assert y[1].item() == 0.0
    assert y[2].item() == 1.0
    assert y[0].item() < 0.0
