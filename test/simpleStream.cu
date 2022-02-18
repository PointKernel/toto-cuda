#include <iostream>

#include <assert.h>

int main()
{
  cudaStream_t stream;

  cudaStreamCreate(&stream);

  assert(cudaStreamDestroy(stream) == 0 /* destroyed without issues*/);
  assert(cudaStreamDestroy(stream) == 709 /* cudaErrorContextIsDestroyed */);

  // Default stream is non-owning thus should not be destroyed by users
  assert(cudaStreamDestroy(0 /*default stream*/) == 400 /* cudaErrorInvalidResourceHandle */);

  return 0;
}
