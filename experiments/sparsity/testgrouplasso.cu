//Usage: dim3 block(c,1); dim3 thread(1,n); col_group_lasso_kernel<<<block,thread>>>(n,c,x,y);
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>
#include <iostream>
#include <fstream>
using std::cout;
using namespace std;
// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if(error != cudaSuccess) \
    cout << "error " << cudaGetErrorString(error); \
  } while (0)

template  <typename Dtype>
__global__ void col_group_lasso_kernel(const int n, const int c, const Dtype *x, Dtype* y){
    int n_offset = 0;
        //initialize y
        while(n_offset<n){
            //int idx1 = (n_offset+threadIdx.y)*gridDim.x+blockIdx.x;
            int idx1 = (n_offset+threadIdx.y)*c+blockIdx.x;
            if(n_offset+threadIdx.y < n){//BUG: THE N MUST BE MULTIPLE TIMES OF BLOCKDIM.Y IN CURRENT IMPLEMENTATION !!!
                y[idx1] = x[idx1]*x[idx1];
            }
            n_offset += blockDim.y;
        }
        __syncthreads();

        //sum along columns
        n_offset=0;
        Dtype res = 0;
        while(n_offset<n){
            int len = (n_offset + blockDim.y)<n ? blockDim.y : (n-n_offset);//valid threads to process
            while(len/2>0){
                if(threadIdx.y<len/2){
                    //int idx1 = (n_offset+threadIdx.y)*gridDim.x+blockIdx.x;
                    //int idx2 = (n_offset+threadIdx.y+(len+1)/2)*gridDim.x+blockIdx.x;
                    int idx1 = (n_offset+threadIdx.y)*c+blockIdx.x;
                    int idx2 = (n_offset+threadIdx.y+(len+1)/2)*c+blockIdx.x;
                    y[idx1] += y[idx2];
                }
                __syncthreads();
                len=(len+1)/2;
            }

            //res += y[n_offset*gridDim.x+blockIdx.x];
            res += y[n_offset*c+blockIdx.x];
            n_offset += blockDim.y;
        }
        __syncthreads();

        //copy
        n_offset=0;
        while(n_offset<n){
            //int idx1 = (n_offset+threadIdx.y)*gridDim.x+blockIdx.x;
            int idx1 = (n_offset+threadIdx.y)*c + blockIdx.x;
            if(n_offset+threadIdx.y < n){
                if(res){
                    y[idx1] = Dtype(sqrt(res));
                }else{
                    y[idx1] = Dtype(0);
                }
            }
              n_offset += blockDim.y;
        }
}

//Usage: dim3 block(1,n); dim3 thread(c,1); row_group_lasso_kernel<<<block,thread>>>(n,c,x,y);
template  <typename Dtype>
__global__ void row_group_lasso_kernel(const int n, const int c, const Dtype *x, Dtype* y){
    int c_offset = 0;

        //initialize y
        while(c_offset<c){
            //int idx1 = blockIdx.y * blockDim.x + c_offset + threadIdx.x;
            int idx1 = blockIdx.y * c + c_offset + threadIdx.x;
            if(c_offset + threadIdx.x < c){//WITHOUT THIS: THE C MUST BE MULTIPLE TIMES OF BLOCKDIM.X IN CURRENT IMPLEMENTATION !!!
                y[idx1] = x[idx1]*x[idx1];
            }
            c_offset += blockDim.x;
        }
        __syncthreads();

        //sum along rows
        c_offset=0;
        Dtype res = 0;
        while(c_offset<c){
            int len = (c_offset + blockDim.x)<c ? blockDim.x : (c-c_offset);//valid threads to process
            while(len/2>0){
                if(threadIdx.x<len/2){
                    //int idx1 = blockIdx.y * blockDim.x + c_offset + threadIdx.x;
                    //int idx2 = blockIdx.y * blockDim.x + c_offset + threadIdx.x + (len+1)/2;
                    int idx1 = blockIdx.y * c + c_offset + threadIdx.x;
                    int idx2 = blockIdx.y * c + c_offset + threadIdx.x + (len+1)/2;
                    y[idx1] += y[idx2];
                }
                __syncthreads();
                len=(len+1)/2;
            }

            //res += y[blockIdx.y * blockDim.x + c_offset];
            res += y[blockIdx.y * c + c_offset];
            c_offset += blockDim.x;
        }
        __syncthreads();

        //copy
        c_offset=0;
        while(c_offset<c){
            //int idx1 = blockIdx.y * blockDim.x + c_offset + threadIdx.x;
            int idx1 = blockIdx.y * c + c_offset + threadIdx.x;
            if(c_offset + threadIdx.x < c){
                if(res){
                    y[idx1] = Dtype(sqrt(res));
                }else{
                    y[idx1] = Dtype(0);
                }
            }
              c_offset += blockDim.x;
        }
}

inline static int get_threads_per_block() {
      cudaDeviceProp prop;
      int device;
      if (cudaSuccess != cudaGetDevice(&device)) {
        std::cout<<"No cuda device present.";
        return 512;
      }
      CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
      return prop.maxThreadsPerBlock;
}

void caffe_gpu_bar_group_lasso(const int n, const int c, const float* x, float* y, bool along_column_or_row){
    int threads_per_block = get_threads_per_block();
    cout << "threads_per_block=" << threads_per_block<<"\n";
    //LOG(INFO)<<"threads_per_block "<<threads_per_block;
    if(along_column_or_row){
        dim3 block(c,1);
        dim3 thread(1,n>threads_per_block ? threads_per_block:n );//CAFFE_CUDA_NUM_THREADS
        col_group_lasso_kernel<<<block,thread>>>(n,c,x,y);
    }else{
        dim3 block(1,n);
        dim3 thread(c>threads_per_block ? threads_per_block:c, 1);//CAFFE_CUDA_NUM_THREADS
        row_group_lasso_kernel<<<block,thread>>>(n,c,x,y);
    }
}



#include <stdio.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  int N = 2;//456;
  int C = 3;//4096;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*C*sizeof(float));
  y = (float*)malloc(N*C*sizeof(float));

  cudaMalloc(&d_x, N*C*sizeof(float)); 
  cudaMalloc(&d_y, N*C*sizeof(float));

  for (int i = 0; i < N*C; i++) {
    x[i] = .00001f*i;
    y[i] = 1.1f;
  }

  cudaMemcpy(d_x, x, N*C*sizeof(float), cudaMemcpyHostToDevice);
  saxpy<<<(N*C+255)/256, 256>>>(N*C, 2.0f, d_x, d_y); 
  //caffe_gpu_bar_group_lasso(N, C, d_x, d_y, true);
  CUDA_CHECK(cudaPeekAtLastError());
  
  cudaMemcpy(y, d_y, N*C*sizeof(float), cudaMemcpyDeviceToHost);
  ofstream myfile;
  myfile.open ("groplasso.txt");
  for (int i = 0; i < N*C; i++)
    myfile << y[i] << "\n";
  
  myfile.close();
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}
