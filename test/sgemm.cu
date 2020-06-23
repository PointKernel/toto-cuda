#include <iostream>

const long int IMAGE_SIZE = 4096;
const int BLOCK_SIZE = 16;

const float alpha = 2.f;
const float beta = 2.f;

__global__
void sgemmNaive(float *A, float *B, float *C, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float val = 0.f;
  for (int i  = 0; i < N; i++)
    val += A[row * N + i] * B[i * N + col];
  C[row * N + col] = alpha * val + beta * C[row * N + col];
}

__global__
void sgemmSHM(float *A, float *B, float *C, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float val = 0.f;

  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

  for (int i  = 0; i < N / BLOCK_SIZE; i++) {
    As[threadIdx.y][threadIdx.x] = A[row * N + BLOCK_SIZE * i + threadIdx.x];
    Bs[threadIdx.y][threadIdx.x] = B[(threadIdx.y + i * BLOCK_SIZE) * N + col];
    __syncthreads();
    for (int j = 0; j < BLOCK_SIZE; j++)
      val += As[threadIdx.y][j] * Bs[j][threadIdx.x];
    __syncthreads();
  }
  C[row * N + col] = alpha * val + beta * C[row * N + col];
}

int main() {
  float *A, *A_d, *B, *B_d, *C, *C_d;
  const int data_size = IMAGE_SIZE * IMAGE_SIZE * sizeof(float);

  cudaMallocHost(&A, data_size);
  cudaMallocHost(&B, data_size);
  cudaMallocHost(&C, data_size);
  cudaMalloc(&A_d, data_size);
  cudaMalloc(&B_d, data_size);
  cudaMalloc(&C_d, data_size);

  const int grid_size = IMAGE_SIZE / BLOCK_SIZE;
  dim3 grid(grid_size, grid_size);  // 128 * 128
  dim3 block(BLOCK_SIZE, BLOCK_SIZE); // 32 x 32 = 1024


  for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i ++) {
    A[i] = 1.f;
    B[i] = 1.f;
    C[i] = 1.f;
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMemcpy(A_d, A, data_size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, data_size, cudaMemcpyHostToDevice);
  cudaMemcpy(C_d, C, data_size, cudaMemcpyHostToDevice);

  cudaEventRecord(start);
  sgemmNaive<<<grid, block>>>(A_d, B_d, C_d, IMAGE_SIZE);
  cudaEventRecord(stop);

  cudaMemcpy(C, C_d, data_size, cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);

  // runtime and FLOP rate
  float milliseconds = 0.f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  double seconds = static_cast<double>(milliseconds) / 1000.;
  std::cout << "sgemmNaive runtime: " << seconds << "\n";
  std::cout << "Performance (TFLOPS/s): "
       << (IMAGE_SIZE * IMAGE_SIZE * IMAGE_SIZE) * 2.0 / seconds / 1e12 << "\n\n";

  for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i ++) {
    A[i] = 1.f;
    B[i] = 1.f;
    C[i] = 1.f;
  }

  cudaEvent_t start1, stop1;
  cudaEventCreate(&start1);
  cudaEventCreate(&stop1);

  cudaMemcpy(A_d, A, data_size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, data_size, cudaMemcpyHostToDevice);
  cudaMemcpy(C_d, C, data_size, cudaMemcpyHostToDevice);

  cudaEventRecord(start1);
  sgemmSHM<<<grid, block>>>(A_d, B_d, C_d, IMAGE_SIZE);
  cudaEventRecord(stop1);

  cudaMemcpy(C, C_d, data_size, cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop1);

  // runtime and FLOP rate
  milliseconds = 0.f;
  cudaEventElapsedTime(&milliseconds, start1, stop1);
  seconds = static_cast<double>(milliseconds) / 1000.;
  std::cout << "sgemmSHM runtime: " << seconds << "\n";
  std::cout << "Performance (TFLOPS/s): "
       << (IMAGE_SIZE * IMAGE_SIZE * IMAGE_SIZE) * 2.0 / seconds / 1e12 << "\n\n";

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  cudaFreeHost(A);
  cudaFreeHost(B);
  cudaFreeHost(C);
}
