#include <iostream>

#include <helper_cuda.h>

using namespace std;

__global__ void kernel(float *a, int offset) {
  int i = offset + threadIdx.x + blockIdx.x * blockDim.x;
  float x = (float)i;
  float s = sinf(x);
  float c = cosf(x);
  a[i] = a[i] + sqrtf(s * s + c * c);
}

int main(int argc, char **argv) {
  const int blockSize = 256, nStreams = 4;
  const int n = 1024;
  const int streamSize = n / nStreams;
  const int streamBytes = streamSize * sizeof(float);
  const int bytes = n * sizeof(float);

  // allocate pinned host memory and device memory
  float *a, *d_a;
  checkCudaErrors(cudaMallocHost((void **)&a, bytes)); // host pinned
  checkCudaErrors(cudaMalloc((void **)&d_a, bytes));   // device

  // create events and streams
  cudaEvent_t start, stop;
  cudaStream_t stream[nStreams];
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  for (int i = 0; i < nStreams; ++i)
    checkCudaErrors(cudaStreamCreate(&stream[i]));

  memset(a, 0, bytes);

  checkCudaErrors(cudaEventRecord(start));
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * streamSize;
    checkCudaErrors(cudaMemcpyAsync(&d_a[offset], &a[offset], streamBytes,
                                    cudaMemcpyHostToDevice, stream[i]));
    kernel<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
    checkCudaErrors(cudaMemcpyAsync(&a[offset], &d_a[offset], streamBytes,
                                    cudaMemcpyDeviceToHost, stream[i]));
  }
  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

  float milliseconds = 0.f;
  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

  cout << "Runtime: " << milliseconds << " ms\n";

  // cleanup
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  for (int i = 0; i < nStreams; ++i)
    checkCudaErrors(cudaStreamDestroy(stream[i]));
  cudaFree(d_a);
  cudaFreeHost(a);

  return 0;
}
