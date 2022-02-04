#include <iostream>

int main() {
  cudaStream_t stream;

  cudaStreamCreate(&stream);

  std::cout << cudaStreamDestroy(stream) << "\n";
  std::cout << cudaStreamDestroy(stream) << "\n";
  std::cout << cudaStreamDestroy(stream) << "\n";

  std::cout << cudaStreamDestroy(cudaStreamDefault) << "\n";

  return 0;
}
