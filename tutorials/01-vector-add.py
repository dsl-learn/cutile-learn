import torch

import cuda.tile as ct
import torch
import math
import triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# Define a type alias for Constant integers.
# This helps in clearly indicating that certain kernel parameters are compile-time constants.
ConstInt = ct.Constant[int]


# --- Kernel 1: 1D Tiled Vector Add (Direct Load/Store) ---
@ct.kernel
def vec_add_kernel_1d(a, b, c, TILE: ConstInt):
    """
    cuTile kernel for 1D element-wise vector addition using direct tiled loads/stores.

    Each thread block processes a `TILE`-sized chunk of the vectors.
    This approach is efficient when the total dimension is a multiple of `TILE`,
    or when out-of-bounds accesses are implicitly handled by the calling context
    (e.g., by padding or ensuring input sizes match grid dimensions).

    Args:
        a: Input tensor A.
        b: Input tensor B.
        c: Output tensor for the sum (A + B).
        TILE (ConstInt): The size of the tile (chunk of data) processed by each
                         thread block. This must be a compile-time constant.
    """
    # Get the global ID of the current thread block along the first dimension.
    # In a 1D grid, this directly corresponds to the index of the tile.
    bid = ct.bid(0)

    # Load TILE-sized chunks from input vectors 'a' and 'b'.
    # `ct.load` automatically distributes the load operation across the threads
    # within the block, bringing the specified tile of data into shared memory
    # or registers. The `index=(bid,)` specifies which tile to load based on the block ID.
    a_tile = ct.load(a, index=(bid,), shape=(TILE,))
    b_tile = ct.load(b, index=(bid,), shape=(TILE,))

    # Perform the element-wise addition on the loaded tiles.
    # This operation happens in parallel across the threads within the block.
    sum_tile = a_tile + b_tile

    # Store the resulting TILE-sized chunk back to the output vector 'c'.
    # `ct.store` writes the computed tile back to global memory, again
    # distributing the store operation across threads.
    ct.store(c, index=(bid,), tile=sum_tile)


def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    N = output.numel()
    # Heuristic for TILE size:
    # Choose a power of 2, up to 1024, that is greater than or equal to N.
    # This helps in efficient memory access patterns on the GPU.
    # Handle N=0 gracefully to avoid log2(0) errors.
    TILE = min(1024, 2 ** math.ceil(math.log2(N))) if N > 0 else 1

    # Calculate the grid dimensions for launching the kernel.
    # `math.ceil(N / TILE)` determines the number of blocks needed to cover
    # the entire vector. Each block processes a `TILE`-sized chunk.
    grid = (math.ceil(N / TILE), 1, 1)  # (blocks_x, blocks_y, blocks_z)

    ct.launch(torch.cuda.current_stream(), grid, vec_add_kernel_1d, (x, y, output, TILE))
    return output


# %%
# We can now use the above function to compute the element-wise sum of two `torch.tensor` objects and test its correctness:

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_cutile = add(x, y)
print(output_torch)
print(output_cutile)
print(f'The maximum difference between torch and cutile is '
      f'{torch.max(torch.abs(output_torch - output_cutile))}')
