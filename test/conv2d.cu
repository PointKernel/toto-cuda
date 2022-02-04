#include <iostream>

const int image_size  = 4096;
const int filter_size = 3;

__global__ void conv2d(int* A, int* B, int* C, int N, int n)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  const int offset = n / 2;

  int row_i = threadIdx.y - offset;
  int col_i = threadIdx.x - offset;

  __shared__ int shm[16][16];

  shm[threadIdx.y][threadIdx.x] = A[row * N + col];

  __syncthreads();

  int val = 0;

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      if ((0 <= (i + col_i) && (i + col_i) < 16))
        if ((0 <= (j + row_i) && (j + row_i) < 16)) val += shm[j + row_i][i + col_i] * C[j * n + i];

  B[row * N + col] = val;
}

int main()
{
  int *A, *A_d, *B, *B_d, *C, *C_d;
  const int data_size   = image_size * image_size * sizeof(int);
  const int kernel_size = filter_size * filter_size * sizeof(int);

  cudaMallocHost(&A, data_size);
  cudaMallocHost(&B, data_size);
  cudaMallocHost(&C, kernel_size);

  for (int i = 0; i < image_size * image_size; i++)
    A[i] = 1;
  memset(B, 0, data_size);
  for (int i = 0; i < filter_size * filter_size; i++)
    C[i] = 2;

  cudaMalloc(&A_d, data_size);
  cudaMalloc(&B_d, data_size);
  cudaMalloc(&C_d, kernel_size);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMemcpy(A_d, A, data_size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, data_size, cudaMemcpyHostToDevice);
  cudaMemcpy(C_d, C, kernel_size, cudaMemcpyHostToDevice);

  const int block_size = 16;
  const int grid_size  = (image_size + block_size - 1) / block_size;
  dim3 grid(grid_size, grid_size);
  dim3 block(block_size, block_size);

  cudaEventRecord(start);
  conv2d<<<grid, block>>>(A_d, B_d, C_d, image_size, filter_size);
  cudaEventRecord(stop);

  cudaMemcpy(B, B_d, data_size, cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++)
      std::cout << B[i * image_size + j] << " ";
    std::cout << "\n";
  }
  std::cout << "Kernel run time: " << milliseconds << " ms\n";

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  cudaFreeHost(A);
  cudaFreeHost(B);
  cudaFreeHost(C);
}
