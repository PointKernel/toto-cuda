# toto-cuda
[![License:MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/l0icenses/MIT)
[![Build Status](https://travis-ci.org/PointKernel/toto-cuda.svg?branch=master)](https://travis-ci.org/PointKernel/toto-cuda)

Benchmarks and unit tests for GPUs

## analysis
Roofline Analysis with Nsight 

## benchmark
1. [ ] `simpleGEMM`: CUBLAS GEMM kernel to measure FLOP rate on Tensor Core
2. [ ] `simpleOverhead`: measure kernel launch overhead on GPUs

## polymorphism
OMP example of polymorphism on GPUs

## pycuda
Simple pycuda kernels with Nsight profiling script

## script
1. [ ] `run_gemms.sh`: loop over GEMMs to measure Tensor Core performance

## test
1. [ ] `conv2d`: 2D convolution kernel using shared memory
2. [ ] `sgemm`: performance improvement via shared memory for GEMM
3. [ ] `simpleOverlap`: code example using CUDA streams
4. [ ] `simpleShuffle`: simple use case of shuffle operations
5. [ ] `thrustVector`: use thrust vector with CUDA
