import torch

import cuda.tile as ct
import torch
import math
import triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# Type aliases for constants
ConstInt = ct.Constant[int]

def swizzle_2d(M, N, TILE_SIZE_M, TILE_SIZE_N, GROUP_SIZE_M):
    # Get the global IDs of the current CUDA block (CTA) in a 1D grid.
    bid = ct.bid(0)
    num_bid_m = ct.cdiv(M, TILE_SIZE_M)
    num_bid_n = ct.cdiv(N, TILE_SIZE_N)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2))
def matmul_kernel(A, B, C,
                  TILE_SIZE_M: ConstInt,         # Tile size along M dimension (rows of C)
                  TILE_SIZE_N: ConstInt,         # Tile size along N dimension (columns of C)
                  TILE_SIZE_K: ConstInt):        # Tile size along K dimension (inner product dimension)
    """
    cuTile kernel for performing matrix multiplication C = A @ B.

    This kernel uses a tiled approach, where each CUDA thread block (CTA)
    computes a `TILE_SIZE_M` x `TILE_SIZE_N` tile of the output matrix C. The computation
    involves iterating over the K-dimension in chunks of `TILE_SIZE_K`.

    Args:
        A: Input matrix A (M x K).
        B: Input matrix B (K x N).
        C: Output matrix C (M x N).
        TILE_SIZE_M (ConstInt): The height of the output tile computed by this block.
                       Corresponds to rows of A and C.
        TILE_SIZE_N (ConstInt): The width of the output tile computed by this block.
                       Corresponds to columns of B and C.
        TILE_SIZE_K (ConstInt): The depth of the inner loop (K-dimension) tile size.
                       Corresponds to columns of A and rows of B.
    """
    GROUP_SIZE_M = 8
    M = A.shape[0]
    N = B.shape[1]
    bidx, bidy = swizzle_2d(M, N, TILE_SIZE_M, TILE_SIZE_N, GROUP_SIZE_M)

    # Calculate the total number of K-tiles that need to be processed.
    # `ct.num_tiles(A, axis=1, shape=(TILE_SIZE_M, TILE_SIZE_K))` extracts the K-dimension (axis 1)
    # from matrix A's shape, assuming A's shape is conceptually (M_tiles, K_tiles),
    # and then implicitly performs ceiling division by `TILE_SIZE_K` to get the number of K-tiles.
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(TILE_SIZE_M, TILE_SIZE_K))

    # Initialize an accumulator for the current output tile (TILE_SIZE_M x TILE_SIZE_N).
    # It's common practice to use `float32` for accumulation even with `float16` inputs
    # to maintain higher precision during the sum-reduction of the matrix multiplication.
    accumulator = ct.full((TILE_SIZE_M, TILE_SIZE_N), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    # Convert fp32 to tf32 to use tensorcore
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    # K-dimension loop: Iterate over the K-dimension in chunks of 'TILE_SIZE_K'.
    # In each iteration, a `TILE_SIZE_M` x `TILE_SIZE_K` tile from A and a `TILE_SIZE_K` x `TILE_SIZE_N` tile from B
    # are loaded, multiplied, and accumulated.
    for k in range(num_tiles_k):
        # Load tile from matrix A.
        # The `index=(bidx, k_tile_idx)` specifies which (M-tile, K-tile) to load
        # from global memory A. `shape=(TILE_SIZE_M, TILE_SIZE_K)` defines the size of this tile.
        a = ct.load(A, index=(bidx, k), shape=(TILE_SIZE_M, TILE_SIZE_K), padding_mode=zero_pad).astype(dtype)

        # Load tile from matrix B.
        # The `index=(k_tile_idx, bidy)` specifies which (K-tile, N-tile) to load
        # from global memory B. `shape=(TILE_SIZE_K, TILE_SIZE_N)` defines the size of this tile.
        b = ct.load(B, index=(k, bidy), shape=(TILE_SIZE_K, TILE_SIZE_N), padding_mode=zero_pad).astype(dtype)

        # Perform Matrix Multiplication for the current tiles.
        # `ct.mma` computes the product of the two loaded tiles and accumulates the result.
        accumulator = ct.mma(a, b, accumulator)

    # Convert the final accumulated result to the desired output data type (C.dtype).
    # This might downcast from float32 to float16 if the output is float16.
    accumulator = ct.astype(accumulator, C.dtype)

    # Store the computed tile to the global memory of the output matrix C.
    # The `(bidx, bidy)` directly corresponds to the tile's position in the 2D output matrix.
    ct.store(C, index=(bidx, bidy), tile=accumulator)

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from autotuner import Autotuner, Config, autotune

def _matmul_autotune_configs():

    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability in [(12, 0), (12, 1)]:
        # sm120, sm121
        configs = [
            Config(TILE_SIZE_M=128, TILE_SIZE_N=64, TILE_SIZE_K=64, num_ctas=1, occupancy=1),
            Config(TILE_SIZE_M=128, TILE_SIZE_N=64, TILE_SIZE_K=32, num_ctas=1, occupancy=2),
        ]
    else:
        # sm100 (Blackwell)
        configs = [
            Config(TILE_SIZE_M=128, TILE_SIZE_N=128, TILE_SIZE_K=32, num_ctas=1, occupancy=1),
            Config(TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=64, num_ctas=2, occupancy=1),
            Config(TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=64, num_ctas=4, occupancy=1),
            Config(TILE_SIZE_M=512, TILE_SIZE_N=256, TILE_SIZE_K=64, num_ctas=2, occupancy=1),
        ]
    return configs

@autotune(search_space=_matmul_autotune_configs())
def matmul(a, b, autotuner: Autotuner | None = None):
    # Check constraints.
    # assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    from math import ceil
    tuned_result = autotuner(
        torch.cuda.current_stream(),
        grid_fn=lambda named_args, cfg: (
            ceil(M / cfg.TILE_SIZE_M) * ceil(N / cfg.TILE_SIZE_N),
            1,
            1,
        ),
        kernel=matmul_kernel,
        args_fn=lambda cfg: (a, b, c, cfg.TILE_SIZE_M, cfg.TILE_SIZE_N, cfg.TILE_SIZE_K),
    )
    return c

# %%
# Unit Test
# ---------
#
# We can test our custom matrix multiplication operation against a native torch implementation (i.e., cuBLAS).

torch.manual_seed(0)
a = torch.rand((512, 512), device=DEVICE, dtype=torch.float16) - 0.5
b = torch.rand((512, 512), device=DEVICE, dtype=torch.float16) - 0.5
cutile_output = matmul(a, b)
torch_output = torch.matmul(a, b)
print(f"cutile_output_with_fp16_inputs={cutile_output}")
print(f"torch_output_with_fp16_inputs={torch_output}")

if torch.allclose(cutile_output, torch_output, atol=1e-2, rtol=0):
    print("✅ cuTile and Torch match")
else:
    print("❌ cuTile and Torch differ")


TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
if TORCH_HAS_FP8:
    torch.manual_seed(0)
    a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    a = a.to(torch.float8_e5m2)
    # pre-transpose b for efficiency.
    b = b.T
    b = b.to(torch.float8_e5m2)
    cutile_output = matmul(a, b)
    torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
    print(f"cutile_output_with_fp8_inputs={cutile_output}")
    print(f"torch_output_with_fp8_inputs={torch_output}")
    if torch.allclose(cutile_output, torch_output, atol=0.125, rtol=0):
        print("✅ cuTile and Torch match")
    else:
        print("❌ cuTile and Torch differ")

# %%
# Benchmark
# ---------
#
# Square Matrix Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can now compare the performance of our kernel against that of cuBLAS. Here we focus on square matrices,
# but feel free to arrange this script as you wish to benchmark any other matrix shape.

ref_lib = 'cuBLAS'

configs = []
for fp8_inputs in [False, True]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["cutile", "triton"] if fp8_inputs else ["cutile", "triton", ref_lib.lower()],  # Label name for the lines
            line_names=["cuTile", "Triton"] if fp8_inputs else ["cuTile", "Triton", ref_lib],  # Line styles
            styles=[("orange", "-"), ("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-" +
            ("fp16" if not fp8_inputs else "fp8"),  # Name for the plot, used also as a file name for saving the plot.
            args={"fp8_inputs": fp8_inputs},
        ))

from triton_kernels import matmul as triton_mutmul

@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    if TORCH_HAS_FP8 and fp8_inputs:
        a = a.to(torch.float8_e5m2)
        b = b.T
        b = b.to(torch.float8_e5m2)
    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_mutmul(a, b), quantiles=quantiles)
    if provider == 'cutile':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

benchmark.run(show_plots=True, print_data=True)
