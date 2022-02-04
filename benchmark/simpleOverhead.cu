#include <helper_cuda.h>
#include <iostream>

__global__ void emptyKernel() {}

int main()
{
  const int N = 100000;
  float time, total = 0.f;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int j = 1; j <= 4096; j *= 2) {
    for (int k = 1; k <= 1024; k *= 2) {
      for (int i = 0; i < N; i++) {
        cudaEventRecord(start, 0);
        emptyKernel<<<1, 1>>>();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        total = total + time;
      }
      std::cout << "Kernel: " << j << "X" << k << "\tlaunch overhead: " << total / N * 1000
                << " us\n";
      total = 0.f;
    }
  }

  total = 0.f;

  void* dst = nullptr;
  void* src = nullptr;
  for (int i = 0; i < N; i++) {
    cudaEventRecord(start, 0);
    checkCudaErrors(cudaMemcpy(dst, src, 0, cudaMemcpyDefault));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    total = total + time;
  }

  std::cout << "\nData transfer overhead: " << total / N * 1000 << " us\n";

  return 0;
}
