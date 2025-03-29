import torch
import torch_tensorrt as trt
import time

from model import *


def compile(load_series):
    model = ValueNet(if_Pacman=True)
    load_name = f"model/pacman_{load_series}.pth"
    model.load_state_dict(torch.load(load_name, map_location=device, weights_only=True))
    trt_model = trt.compile(model, inputs=[torch.randn(1, 2, 42, 42).cuda()], enabled_precisions={torch.float16})
    torch.jit.save(trt_model, f"jitmodel/pacman_{load_series}.pt")

def test(load_series):
    model = torch.jit.load(f"jitmodel/pacman_{load_series}.pt").cuda()
    model.eval()
    input_tensor = torch.randn(1, 2, 42, 42).cuda()
    t = time.time()
    output = model(input_tensor)
    t = time.time()-t
    print(output, t)

COMPILE = True
TEST = True

if __name__ == "__main__":
    load_series = "03290333"
    
    if COMPILE:
        compile(load_series)
    if TEST:
        test(load_series)