#include <iostream>

int main()
{
  cudaStream_t stream;

  cudaStreamCreate(&stream);

  std::cout << cudaStreamDestroy(stream) << "\n";
  std::cout << cudaStreamDestroy(stream) << "\n";
  std::cout << cudaStreamDestroy(stream) << "\n";

  return 0;
}
