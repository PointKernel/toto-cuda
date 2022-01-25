#include <helper_cuda.h>
#include <iostream>

#define FULL_MASK 0xffffffff

__global__ void shfl_up_kernel(int32_t* array_d)
{
  if (threadIdx.x < 256) {
    auto x      = array_d[threadIdx.x];
    auto laneId = threadIdx.x & 0x1f;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      auto y = __shfl_up_sync(FULL_MASK, x, offset);
      if (laneId >= offset) x += y;
    }
    array_d[threadIdx.x] = x;
  }
}

__global__ void shfl_down_kernel(int* array_d)
{
  int val = 1;
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_MASK, val, offset);
  if (threadIdx.x == 0) array_d[blockIdx.x] = val;
}

int main()
{
  int *array_h, *array_d;

  array_h = static_cast<int*>(malloc(sizeof(int) * 32));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&array_d), sizeof(int) * 32));

  shfl_down_kernel<<<32, 32>>>(array_d);
  cudaMemcpy(reinterpret_cast<void*>(array_h),
             reinterpret_cast<void*>(array_d),
             32 * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  std::cout << array_h[0] << "\n\n";

  int32_t *a_h, *a_d;

  const int N = 512;
  a_h         = static_cast<int32_t*>(malloc(sizeof(int32_t) * N));
  for (int i = 0; i < N; i++)
    a_h[i] = i;

  checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&a_d), sizeof(int32_t) * N));

  cudaMemcpy(reinterpret_cast<void*>(a_d),
             reinterpret_cast<void*>(a_h),
             sizeof(int32_t) * N,
             cudaMemcpyHostToDevice);
  shfl_up_kernel<<<1, 512>>>(a_d);
  cudaMemcpy(reinterpret_cast<void*>(a_h),
             reinterpret_cast<void*>(a_d),
             sizeof(int32_t) * N,
             cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  for (int i = 0; i < N / 32; i++) {
    for (int j = 0; j < 32; j++) {
      std::cout << a_h[i * 32 + j] << " ";
    }
    std::cout << "\n";
  }
  return 0;
}
