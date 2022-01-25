#include <thrust/device_vector.h>

#include <cuda/std/chrono>

#include <iostream>

namespace detail {
// TODO: Use chrono::utc_clock when available in libcu++?
template <class Duration>
using time_point = cuda::std::chrono::sys_time<Duration>;

template <class Duration>
using timestamp = time_point<Duration>;
}  // namespace detail

/**
 * @brief Type alias representing an int64_t duration of seconds.
 */
using duration_s = cuda::std::chrono::duration<int64_t, cuda::std::chrono::seconds::period>;
/**
 * @brief Type alias representing an int64_t duration of milliseconds.
 */
using duration_ms = cuda::std::chrono::duration<int64_t, cuda::std::chrono::milliseconds::period>;
/**
 * @brief Type alias representing an int64_t duration of microseconds.
 */
using duration_us = cuda::std::chrono::duration<int64_t, cuda::std::chrono::microseconds::period>;

/**
 * @brief Type alias representing an int64_t duration of seconds since the
 * unix epoch.
 */
using timestamp_s = detail::timestamp<cuda::std::chrono::duration<int64_t, cuda::std::ratio<1>>>;
/**
 * @brief Type alias representing an int64_t duration of milliseconds since
 * the unix epoch.
 */
using timestamp_ms = detail::timestamp<cuda::std::chrono::duration<int64_t, cuda::std::milli>>;
/**
 * @brief Type alias representing an int64_t duration of microseconds since
 * the unix epoch.
 */
using timestamp_us = detail::timestamp<cuda::std::chrono::duration<int64_t, cuda::std::micro>>;

__global__ void kernel(float* A, int N, float* res)
{
  using namespace cuda::std::chrono;

  duration_s s{1};
  printf("### %ld\n", long(cuda::std::chrono::duration_cast<duration_us>(s).count()));

  extern __shared__ float shm[];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  shm[tid] = A[idx];
  __syncthreads();

  for (int i = 1; i < blockDim.x; i *= 2) {
    int index = 2 * i * tid;
    if (index < blockDim.x) shm[index] += shm[index + i];
    __syncthreads();
  }

  if (tid == 0) atomicAdd(res, shm[0]);
}

int main()
{
  thrust::device_vector<float> d_vec(128, 1.f);
  float* raw_ptr = thrust::raw_pointer_cast(d_vec.data());

  float* d_res;
  float res = 0.f;
  cudaMalloc(&d_res, sizeof(float));
  cudaMemset((void*)d_res, 0, sizeof(float));

  kernel<<<1, 1, 128 * sizeof(float)>>>(raw_ptr, d_vec.size(), d_res);
  cudaMemcpy(&res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << res << "\n";

  std::cout << std::chrono::time_point<std::chrono::system_clock, std::chrono::seconds>::rep;
  std::cout << std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>::rep;
  std::cout << std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds>::rep;

  return 0;
}
