#include <thrust/device_vector.h>

#include <iostream>

__global__ void kernel(float *A, int N, float *res) {
  extern __shared__ float shm[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  shm[tid] = A[idx];
  __syncthreads();

  for (int i = 1; i < blockDim.x; i *= 2) {
    int index = 2 * i * tid;
    if (index < blockDim.x)
      shm[index] += shm[index + i];
    __syncthreads();
  }

  if (tid == 0)
    atomicAdd(res, shm[0]);
}

int main() {
  thrust::device_vector<float> d_vec(128, 1.f);
  float *raw_ptr = thrust::raw_pointer_cast(d_vec.data());

  float *d_res;
  float res = 0.f;
  cudaMalloc(&d_res, sizeof(float));
  cudaMemset((void *)d_res, 0, sizeof(float));

  kernel<<<1, 128, 128 * sizeof(float)>>>(raw_ptr, d_vec.size(), d_res);
  cudaMemcpy(&res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << res << "\n";

  return 0;
}
