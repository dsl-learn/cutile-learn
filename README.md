## cuTile learning

### tutorials

- [01-vector-add](./tutorials/01-vector-add.py)

- [03-matmul](./tutorials/03-matrix-multiplication.py)

- [06-attention](./tutorials/06-fused-attention.py)

### some documents

- [cuTile documents](https://docs.nvidia.com/cuda/cutile-python)
- [Tile IR documents](https://docs.nvidia.com/cuda/tile-ir)
- [PyTorch_Conference_CudaTileIR.pdf](https://static.sched.com/hosted_files/pytorchconference/76/Jared_Roesch_PyTorch_Conference_CudaTileIR_v2.pdf)

### Github repositorys

[NVIDIA/cutile-python](https://github.com/NVIDIA/cutile-python)

[NVIDIA/TileGym](https://github.com/NVIDIA/TileGym)

### YouTube videoes

[![Deep Dive: How to Use cuTile Python](https://img.youtube.com/vi/YFrP03KuMZ8/maxresdefault.jpg)](https://www.youtube.com/watch?v=YFrP03KuMZ8)

[![THE FUTURE IS TILED: using cuTile and CUDA Tile IR to write portable, high-performance GPU Kernels](https://img.youtube.com/vi/UEdGJGz8Eyg/maxresdefault.jpg)](https://www.youtube.com/watch?v=UEdGJGz8Eyg)

## benchmark

Torch 2.9.1, Triton 3.5.1, cuTile(cuda-tile) 1.0.0, tileiras Cuda compilation tools, release 13.1, V13.1.80.

### 5090 FP16 matmul

data in [benchmark/5090/matmul-performance-5090-fp16.csv](benchmark/5090/matmul-performance-5090-fp16.csv)

![matmul-performance-5090-fp16](https://img2024.cnblogs.com/blog/1154439/202512/1154439-20251206094800062-949547188.png)

### 5090 FP8 matmul

data in [benchmark/5090/matmul-performance-5090-fp8.csv](benchmark/5090/matmul-performance-5090-fp8.csv)

![matmul-performance-5090-fp8](https://img2024.cnblogs.com/blog/1154439/202512/1154439-20251206094849117-1333188640.png)
