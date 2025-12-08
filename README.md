# cuTile learning

## Tutorials

- [Vector Addition](./tutorials/01-vector-add.py)

- [Fused Softmax](./tutorials/02-fused-softmax.py)

- [Matrix Multiplication](./tutorials/03-matrix-multiplication.py)

- [Low-Memory Dropout](./tutorials/04-low-memory-dropout.py)

- [Layer Normalization](./tutorials/05-layer-norm.py)

- [Fused Attention](./tutorials/06-fused-attention.py)

## Benchmark

All benchmarks were run with Torch 2.9.1, Triton 3.5.1, cuTile (cuda-tile) 1.0.0, and tileiras, using CUDA compilation tools 13.1 (V13.1.80).

Currently, I only have results from an RTX 5090 (sm_120). Contributions from Blackwell B200 (sm_100) users are very welcome!

### 5090 attention fwd

data in [benchmark/5090/attn](benchmark/5090/attn)

![5090 attention](https://img2024.cnblogs.com/blog/1154439/202512/1154439-20251206183251106-1611398145.png)

### 5090 softmax

data in [benchmark/5090/softmax](benchmark/5090/softmax)

![softmax-performance](https://img2024.cnblogs.com/blog/1154439/202512/1154439-20251208211339603-939997130.png)

### 5090 layer normal

![5090-layer-norm](https://img2024.cnblogs.com/blog/1154439/202512/1154439-20251209070554480-125494017.png)

### 5090 matmul

data in [benchmark/5090/matmul](benchmark/5090/matmul)

![5090 matmul](https://img2024.cnblogs.com/blog/1154439/202512/1154439-20251206182733009-1896350291.png)

## My Zhihu article

[如何评价 cuTile? —— BobHuang的回答](https://www.zhihu.com/question/1980278886894957587/answer/1980592292936062307)

[浅析cuTile执行流程](https://zhuanlan.zhihu.com/p/1981136316507890723)

## Documents

- [cuTile documents](https://docs.nvidia.com/cuda/cutile-python)
- [Tile IR documents](https://docs.nvidia.com/cuda/tile-ir)
- [PyTorch_Conference_CudaTileIR.pdf](https://static.sched.com/hosted_files/pytorchconference/76/Jared_Roesch_PyTorch_Conference_CudaTileIR_v2.pdf)

## Github repositorys

[NVIDIA/cutile-python](https://github.com/NVIDIA/cutile-python)

[NVIDIA/TileGym](https://github.com/NVIDIA/TileGym)

## YouTube videoes

[![Deep Dive: How to Use cuTile Python](https://img.youtube.com/vi/YFrP03KuMZ8/maxresdefault.jpg)](https://www.youtube.com/watch?v=YFrP03KuMZ8)

[![THE FUTURE IS TILED: using cuTile and CUDA Tile IR to write portable, high-performance GPU Kernels](https://img.youtube.com/vi/UEdGJGz8Eyg/maxresdefault.jpg)](https://www.youtube.com/watch?v=UEdGJGz8Eyg)
