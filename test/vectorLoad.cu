#include <thrust/device_vector.h>

template <typename PairType>
__device__ void load_pair_array(PairType *arr, PairType *pair_ptr) {
  if constexpr (sizeof(PairType) == 4) {
    auto const tmp = *reinterpret_cast<ushort4 const *>(pair_ptr);
    memcpy(&arr[0], &tmp, 2 * sizeof(PairType));
  } else {
    auto const tmp = *reinterpret_cast<uint4 const *>(pair_ptr);
    memcpy(&arr[0], &tmp, 2 * sizeof(PairType));
  }
}

template <typename PairType, typename OutputIt>
__global__ void kernel(PairType *array, OutputIt output, const int n) {
  int tid = threadIdx.x;
  auto idx = tid - tid % 2;
  PairType arr[2];
  load_pair_array(&arr[0], array + idx);
  *(output + tid) = arr[0].first + arr[1].first;
}

int main() {
  auto constexpr N = 128;

  std::vector<thrust::pair<int, int>> h_pairs(N);
  for (auto i = 0; i < N; ++i) {
    h_pairs[i].first = h_pairs[i].second = i;
  }
  thrust::device_vector<thrust::pair<int, int>> d_pairs(h_pairs);
  auto input_begin = d_pairs.data().get();

  thrust::device_vector<int> d_results(N);
  auto output_begin = d_results.data().get();

  kernel<<<1, N>>>(input_begin, output_begin, N);

  return 0;
}
