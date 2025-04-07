#include <cstdlib>
#include <stdio.h>
#include <time.h>
#include <cuda.h>

__global__ void non_tiled_stencil(float* input, float* result, int n) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  //result[(z * n * n) + (y * n) + x] = 0;
  if (x > 0 && x < n - 1 && y > 0 && y < n - 1 && z > 0 && z < n - 1) {
    float local_result = 0.0f;
    local_result += input[(z * n * n) + (y * n) + (x - 1)];
    local_result += input[(z * n * n) + (y * n) + (x + 1)];
    local_result += input[(z * n * n) + ((y - 1) * n) + x];
    local_result += input[(z * n * n) + ((y + 1) * n) + x];
    local_result += input[((z - 1) * n * n) + (y * n) + x];
    local_result += input[((z + 1) * n * n) + (y * n) + x];
    result[(z * n * n) + (y * n) + x] = local_result * 0.75f;
  } 
}

__device__ int idx_helper(int x, int y, int z, int n) {
  return (z * n * n) + (y * n) + x;
}

__global__ void tiled_stencil(float* input, float* result, int n, int M) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  
  if (x >= n || y >= n || z >= n) return;
  
  __shared__ float tile[16][16][16]; // Adjust size as needed
  
  int local_x = threadIdx.x + 1;
  int local_y = threadIdx.y + 1;
  int local_z = threadIdx.z + 1;
  int global_idx = idx_helper(x, y, z, n);
  
  tile[local_x][local_y][local_z] = input[global_idx];
  
  // Compute the outer faces
  if (threadIdx.x == 0 && x > 0) 
      tile[0][local_y][local_z] = input[idx_helper(x - 1, y, z, n)];
  if (threadIdx.x == M - 1 && x < n - 1) 
      tile[M + 1][local_y][local_z] = input[idx_helper(x + 1, y, z, n)];
  
  if (threadIdx.y == 0 && y > 0) 
      tile[local_x][0][local_z] = input[idx_helper(x, y - 1, z, n)];
  if (threadIdx.y == M - 1 && y < n - 1) 
      tile[local_x][M + 1][local_z] = input[idx_helper(x, y + 1, z, n)];
  
  if (threadIdx.z == 0 && z > 0) 
      tile[local_x][local_y][0] = input[idx_helper(x, y, z - 1, n)];
  if (threadIdx.z == M - 1 && z < n - 1) 
      tile[local_x][local_y][M + 1] = input[idx_helper(x, y, z + 1, n)];

  __syncthreads();
  
  // Assign shared memory indices to their global indices
  if (x > 0 && x < n - 1 && y > 0 && y < n - 1 && z > 0 && z < n - 1) {
      result[global_idx] = 
          (tile[local_x - 1][local_y][local_z] +
           tile[local_x + 1][local_y][local_z] +
           tile[local_x][local_y - 1][local_z] +
           tile[local_x][local_y + 1][local_z] +
           tile[local_x][local_y][local_z - 1] +
           tile[local_x][local_y][local_z + 1]) * 0.75f;
  }
}


int main(int argc, char* argv[]) {
  int i;
  float sum = 0;
  int n = atoi(argv[1]);
  int size = n * n * n;
  int num_bytes = size * sizeof(float);
  float* a = (float*) calloc(size, sizeof(float));
  for (i = 0; i < size; i++) {
    a[i] = 1;
  }   

  float* b = (float*) calloc(size, sizeof(float));

  int block_size = 8;
  int grid_dim = n / block_size;
  dim3 blockDim(block_size, block_size, block_size);
  dim3 gridDim(grid_dim, grid_dim, grid_dim);

  // Non tiled approach
  cudaEvent_t start, stop;
  float elapsed_time_ms1;
  cudaEventCreate(&start);
  cudaEventCreate(&stop); 

  cudaEventRecord(start, 0);

  float* a_d;
  float* b_d;
  cudaMalloc((void **) &a_d, num_bytes);
  cudaMalloc((void **) &b_d, num_bytes);

  cudaMemcpy(a_d, a, num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b, num_bytes, cudaMemcpyHostToDevice);

  non_tiled_stencil<<<gridDim, blockDim>>>(a_d, b_d, n);

  cudaMemcpy(b, b_d, num_bytes, cudaMemcpyDeviceToHost);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms1, start, stop);
  printf("Time spent: %f\n", elapsed_time_ms1);
  for (i = 0; i < size; i++) {
    sum += b[i];
  }
  printf("Sum for check: %f\n", sum);
  free(b);

  // Tiled approach

  cudaEventRecord(start, 0);

  b = (float*) calloc(size, sizeof(float));

  cudaMemcpy(b_d, b, num_bytes, cudaMemcpyHostToDevice);

  int shmem = block_size * block_size * block_size * sizeof(float); 
  tiled_stencil<<<gridDim, blockDim, shmem>>>(a_d, b_d, n, block_size );

  cudaMemcpy(b, b_d, num_bytes, cudaMemcpyDeviceToHost);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms1, start, stop);
  printf("Time spent: %f\n", elapsed_time_ms1);

  sum = 0;
  for (i = 0; i < size; i++) {
    sum += b[i];
  }
  printf("Sum for check: %f\n", sum);

  free(a);
  free(b);
  cudaFree(a_d);
  cudaFree(b_d);

  return 0;
}
