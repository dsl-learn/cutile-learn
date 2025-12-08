import torch
import cuda.tile as ct
import math


@ct.kernel
def dropout_kernel(
    x,
    x_keep,
    output,
    p: ct.Constant[float],
    TILE: ct.Constant[int],
):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid), shape=(TILE,))
    x_keep_tile = ct.load(x_keep, index=(bid), shape=(TILE,))
    output_tile = ct.where(x_keep_tile, x_tile / (1 - p), 0.0)
    ct.store(output, index=(bid,), tile=output_tile)


def dropout(x, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    N = x.numel()
    TILE = 1024
    grid = (math.ceil(N / TILE), 1, 1)  # (blocks_x, blocks_y, blocks_z)

    ct.launch(torch.cuda.current_stream(), grid, dropout_kernel, (x, x_keep, output, p, TILE))
    return output


DEVICE = torch.cuda.current_device()
N = 98432
# Input tensor
x = torch.randn(size=(N, ), device=DEVICE)
# Dropout mask
p = 0.5
x_keep = (torch.rand(size=(N, ), device=DEVICE) > p).to(torch.bool)
cutile_output = dropout(x, x_keep=x_keep, p=p)

torch_output = torch.where(x_keep.bool(), x / (1 - p), torch.zeros_like(x))

if torch.allclose(cutile_output, torch_output, atol=1e-2, rtol=0):
    print("✅ cuTile and Torch match")
else:
    print("❌ cuTile and Torch differ")

N = 10
x = torch.randn(size=(N, ), device=DEVICE)
x_keep = (torch.rand(size=(N, ), device=DEVICE) > p).to(torch.bool)
output = dropout(x, x_keep=x_keep, p=p)
import tabulate

print(tabulate.tabulate([
    ["input"] + x.tolist(),
    ["keep mask"] + x_keep.tolist(),
    ["output"] + output.tolist(),
]))

# Triton provides tl.rand and has seeded_dropout. We need to implement similar functions
# as defined in Triton Language random based on the Philox algorithm:
# https://github.com/triton-lang/triton/blob/main/python/triton/language/random.py
