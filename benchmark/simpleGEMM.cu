#include <cstdlib>
#include <iostream>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <helper_cuda.h>

using namespace std;

void init(half *A, half *B, float *C, size_t m, size_t n, size_t k) {
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < k; j++)
      A[i * k + j] = __float2half(static_cast<float>(rand() % 100));
  for (size_t i = 0; i < k; i++)
    for (size_t j = 0; j < n; j++)
      B[i * n + j] = __float2half(static_cast<float>(rand() % 100));
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < n; j++)
      C[i * n + j] = static_cast<float>(rand() % 100);
}

int main(int argc, char *argv[]) {
  size_t m_global, n_global, k_global;
  if (argc == 2) {
    cout << "\nMatrix size: " << argv[1] << endl;
    m_global = n_global = k_global = atoi(argv[1]);
  } else {
    const size_t size = 4096;
    cout << "\nUsing default matrix size: " << size << endl;
    m_global = n_global = k_global = size;
  }

  // declare host data
  half *A_h;
  half *B_h;
  float *C_h;
  A_h = (half *)malloc(m_global * k_global * sizeof(half));
  B_h = (half *)malloc(k_global * n_global * sizeof(half));
  C_h = (float *)malloc(m_global * n_global * sizeof(float));

  // declare device data
  half *A_d;
  half *B_d;
  float *C_d;
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&A_d),
                             m_global * k_global * sizeof(half)));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&B_d),
                             k_global * n_global * sizeof(half)));
  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&C_d),
                             m_global * n_global * sizeof(float)));

  // initialize host data
  init(A_h, B_h, C_h, m_global, n_global, k_global);

  // copy host data to device
  cudaMemcpy(reinterpret_cast<void *>(A_d), reinterpret_cast<void *>(A_h),
             m_global * k_global, cudaMemcpyHostToDevice);
  cudaMemcpy(reinterpret_cast<void *>(B_d), reinterpret_cast<void *>(B_h),
             k_global * n_global, cudaMemcpyHostToDevice);
  cudaMemcpy(reinterpret_cast<void *>(C_d), reinterpret_cast<void *>(C_h),
             m_global * n_global, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  // create CUDA events for timing measurement
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle);
  cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);

  float alpha = 2.0f;
  float beta = 2.0f;

  // dim3 gridDim;
  // dim3 blockDim;
  for (int i = 0; i < 10; i++) {
    cudaEventRecord(start);
    checkCudaErrors(cublasGemmEx(
        cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m_global, n_global, k_global,
        &alpha, A_d, CUDA_R_16F, m_global, B_d, CUDA_R_16F, k_global, &beta,
        C_d, CUDA_R_32F, m_global, CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
    cudaEventRecord(stop);
  }

  cudaMemcpy(reinterpret_cast<void *>(C_h), reinterpret_cast<void *>(C_d),
             m_global * n_global, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // print kernel runtime
  cudaEventSynchronize(stop);
  float milliseconds = 0.f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  double seconds = static_cast<double>(milliseconds) / 1000.;
  cout << "runtime: " << seconds << endl;
  cout << "Tensor TFLOPS: "
       << (m_global * n_global * k_global) * 2.0 / seconds / 1e12 << endl;

  // free the allocated memory
  free(A_h);
  free(B_h);
  free(C_h);
  cudaFree(reinterpret_cast<void *>(A_d));
  cudaFree(reinterpret_cast<void *>(B_d));
  cudaFree(reinterpret_cast<void *>(C_d));

  return 0;
}
