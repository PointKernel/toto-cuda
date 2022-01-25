#include <iomanip>
#include <iostream>
#include <string>

const long int GLOBAL_SIZE = 1024;
const int TILE_DIM         = 32;
const int BLOCK_ROWS       = 8;
const int NUM_ITERS        = 100;

__global__ void copy(float* A, float* B)
{
  int row = blockIdx.y * blockDim.x + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < blockDim.x; i += blockDim.y)
    B[(row + i) * GLOBAL_SIZE + col] = A[(row + i) * GLOBAL_SIZE + col];
}

__global__ void transposeNaive(float* A, float* B)
{
  int row = blockIdx.y * blockDim.x + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < blockDim.x; i += blockDim.y)
    B[col * GLOBAL_SIZE + row + i] = A[(row + i) * GLOBAL_SIZE + col];
}

__global__ void transposeCoalescedOutput(float* A, float* B)
{
  int row = blockIdx.y * blockDim.x + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < blockDim.x; i += blockDim.y)
    B[(row + i) * GLOBAL_SIZE + col] = A[col * GLOBAL_SIZE + row + i];
}

__global__ void transposeCoalescedSHM(float* A, float* B)
{
  int row = blockIdx.y * blockDim.x + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float tile[TILE_DIM][TILE_DIM];
  for (int i = 0; i < blockDim.x; i += blockDim.y)
    tile[threadIdx.y + i][threadIdx.x] = A[(row + i) * GLOBAL_SIZE + col];
  __syncthreads();

  row = blockIdx.x * blockDim.x + threadIdx.y;
  col = blockIdx.y * blockDim.x + threadIdx.x;
  for (int i = 0; i < blockDim.x; i += blockDim.y)
    B[(row + i) * GLOBAL_SIZE + col] = tile[threadIdx.x][threadIdx.y + i];
}

__global__ void transposeCoalescedOptimal(float* A, float* B)
{
  int row = blockIdx.y * blockDim.x + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];
  for (int i = 0; i < blockDim.x; i += blockDim.y)
    tile[threadIdx.y + i][threadIdx.x] = A[(row + i) * GLOBAL_SIZE + col];
  __syncthreads();

  row = blockIdx.x * blockDim.x + threadIdx.y;
  col = blockIdx.y * blockDim.x + threadIdx.x;
  for (int i = 0; i < blockDim.x; i += blockDim.y)
    B[(row + i) * GLOBAL_SIZE + col] = tile[threadIdx.x][threadIdx.y + i];
}

void initArray(float* A)
{
  for (int i = 0; i < GLOBAL_SIZE * GLOBAL_SIZE; i++)
    A[i] = static_cast<float>(i);
}

void printtArray(float* A)
{
  for (int i = 0; i < GLOBAL_SIZE; i++) {
    for (int j = 0; j < GLOBAL_SIZE; j++)
      std::cout << std::left << std::setw(10) << A[i * GLOBAL_SIZE + j];
    std::cout << "\n";
  }
}

void printSummary(std::string& s, cudaEvent_t& start, cudaEvent_t& stop)
{
  float milliseconds = 0.f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << std::left << std::setw(30) << s;
  std::cout << 2 * GLOBAL_SIZE * GLOBAL_SIZE * sizeof(float) * NUM_ITERS / (milliseconds * 1e6);
  std::cout << "\n";
}

int main()
{
  float *A, *A_d, *B, *B_d;
  const int data_size = GLOBAL_SIZE * GLOBAL_SIZE * sizeof(float);

  cudaMallocHost(&A, data_size);
  cudaMallocHost(&B, data_size);
  cudaMalloc(&A_d, data_size);
  cudaMalloc(&B_d, data_size);

  const int grid_size = GLOBAL_SIZE / TILE_DIM;  // 1024 / 32 = 32
  dim3 grid(grid_size, grid_size);               // 32 * 32
  dim3 block(TILE_DIM, BLOCK_ROWS);              // 32 x 8 = 256

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // matrix copy
  initArray(A);

  cudaMemcpy(A_d, A, data_size, cudaMemcpyHostToDevice);

  cudaEventRecord(start);
  for (int i = 0; i < NUM_ITERS; i++)
    copy<<<grid, block>>>(A_d, B_d);
  cudaEventRecord(stop);

  cudaMemcpy(B, B_d, data_size, cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);

  std::string s = "copy";
  printSummary(s, start, stop);

  // naive transpose
  initArray(A);

  cudaMemcpy(A_d, A, data_size, cudaMemcpyHostToDevice);

  cudaEventRecord(start);
  for (int i = 0; i < NUM_ITERS; i++)
    transposeNaive<<<grid, block>>>(A_d, B_d);
  cudaEventRecord(stop);

  cudaMemcpy(B, B_d, data_size, cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);

  s = "transposeNaive";
  printSummary(s, start, stop);

  // transpose with coalesced memory access for output only
  initArray(A);

  cudaMemcpy(A_d, A, data_size, cudaMemcpyHostToDevice);

  cudaEventRecord(start);
  for (int i = 0; i < NUM_ITERS; i++)
    transposeCoalescedOutput<<<grid, block>>>(A_d, B_d);
  cudaEventRecord(stop);

  cudaMemcpy(B, B_d, data_size, cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);

  s = "transposeCoalescedOutput";
  printSummary(s, start, stop);

  // transpose with coalesced memory access for both input and output + shared
  // memory
  initArray(A);

  cudaMemcpy(A_d, A, data_size, cudaMemcpyHostToDevice);

  cudaEventRecord(start);
  for (int i = 0; i < NUM_ITERS; i++)
    transposeCoalescedSHM<<<grid, block>>>(A_d, B_d);
  cudaEventRecord(stop);

  cudaMemcpy(B, B_d, data_size, cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);

  s = "transposeCoalescedSHM";
  printSummary(s, start, stop);

  // transpose with optimal coalesced memory access (no bank conflict)
  initArray(A);

  cudaMemcpy(A_d, A, data_size, cudaMemcpyHostToDevice);

  cudaEventRecord(start);
  for (int i = 0; i < NUM_ITERS; i++)
    transposeCoalescedOptimal<<<grid, block>>>(A_d, B_d);
  cudaEventRecord(stop);

  cudaMemcpy(B, B_d, data_size, cudaMemcpyDeviceToHost);
  cudaEventSynchronize(stop);

  s = "transposeCoalescedOptimal";
  printSummary(s, start, stop);

  // Free
  cudaFree(A_d);
  cudaFree(B_d);

  cudaFreeHost(A);
  cudaFreeHost(B);
}
