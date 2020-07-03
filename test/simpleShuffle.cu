#include <helper_cuda.h>
#include <iostream>

#define FULL_MASK 0xffffffff

__global__ void shfl_up_kernel(uint32_t *array_d) {
  uint32_t x = 1;
  uint32_t laneId = 0;
  uint32_t i = 0;
  for (int32_t offset = 1; offset < 32; offset <<= 1) {
    auto y = __shfl_up_sync(FULL_MASK, x, offset);
    if (blockIdx.x == 0)
      array_d[i * 32 + threadIdx.x] = y;
    __syncthreads();
    if (laneId >= offset)
      x += y;
    i++;
  }
  if (blockIdx.x == 0)
    array_d[threadIdx.x] = x;
}

__global__ void shfl_down_kernel(int *array_d) {
  int val = 2;
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_MASK, val, offset);
  if (threadIdx.x == 0)
    array_d[blockIdx.x] = val;
}

int main() {
  int *array_h, *array_d;

  array_h = static_cast<int *>(malloc(sizeof(int) * 32));
  checkCudaErrors(
      cudaMalloc(reinterpret_cast<void **>(&array_d), sizeof(int) * 32));

  shfl_down_kernel<<<32, 32>>>(array_d);
  cudaMemcpy(reinterpret_cast<void *>(array_h),
             reinterpret_cast<void *>(array_d), 32, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  std::cout << array_h[0] << "\n\n";

  uint32_t *a_h, *a_d;

  a_h = static_cast<uint32_t *>(malloc(sizeof(uint32_t) * 32 * 6));
  checkCudaErrors(
      cudaMalloc(reinterpret_cast<void **>(&a_d), sizeof(uint32_t) * 32 * 6));

  shfl_up_kernel<<<32, 32>>>(a_d);
  cudaMemcpy(reinterpret_cast<void *>(a_h), reinterpret_cast<void *>(a_d),
             32 * 6, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 32; j++) {
      std::cout << a_h[i * 32 + j] << " ";
    }
    std::cout << "\n";
  }
  return 0;
}
