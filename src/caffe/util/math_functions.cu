#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_geam<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N,
    const float alpha, const float* A, const float beta, const float* B,
    float* C){
	  int lda = (TransA == CblasNoTrans) ? N : M;
	  int ldb = (TransB == CblasNoTrans) ? N : M;
	  cublasOperation_t cuTransA =
	      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	  cublasOperation_t cuTransB =
	      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

	  CUBLAS_CHECK(cublasSgeam(Caffe::cublas_handle(), cuTransA, cuTransB,
		  N, M, &alpha, A, lda, &beta, B, ldb, C, N));
}

template <>
void caffe_gpu_geam<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N,
    const double alpha, const double* A, const double beta, const double* B,
    double* C){
	  int lda = (TransA == CblasNoTrans) ? N : M;
	  int ldb = (TransB == CblasNoTrans) ? N : M;
	  cublasOperation_t cuTransA =
	      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	  cublasOperation_t cuTransB =
	      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

	  CUBLAS_CHECK(cublasDgeam(Caffe::cublas_handle(), cuTransA, cuTransB,
		  N, M, &alpha, A, lda, &beta, B, ldb, C, N));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <typename Dtype>
__global__ void keep_same_direction_kernel(const int n, const Dtype* X, Dtype* Y) {
  CUDA_KERNEL_LOOP(index, n) {
    if(X[index]*Y[index]<0) Y[index] = 0;
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_keep_same_direction<float>(const int N, const float* X,
		float* Y){
  keep_same_direction_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
	      N, X,Y);
}

template <>
void caffe_gpu_keep_same_direction<double>(const int N, const double* X,
		double* Y){
  keep_same_direction_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
	      N, X,Y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <typename Dtype>
__global__ void div_kernel_check_zero(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
	  if(fabs(b[index]) <= ZERO_THRESHOLD) y[index] = 0;
	  else y[index] = a[index] / b[index];
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div_check_zero<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
	div_kernel_check_zero<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div_check_zero<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
	div_kernel_check_zero<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <typename Dtype>
__global__ void powx_kernel_check_negative(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
	if (a[index]<=0) y[index]=0;
	else y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx_check_negative<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel_check_negative<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx_check_negative<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel_check_negative<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}



#define XOFFSET(idx) ((idx)%blk_size_c)
#define YOFFSET(idx) ((idx)/blk_size_c)
//Usage: dim3 block(a,b); dim3 thread(get_threads_per_block,1); block_length_kernel<<<block,thread>>>(n,c,x,y);
//one-D thread block processes two-D data block
template  <typename Dtype>
__global__ void block_length_kernel(const int n, const int c,
										const Dtype *x, Dtype* y){
	int c_offset = 0;
	const int blk_size_n = n%gridDim.y ? n/gridDim.y+1 : n/gridDim.y;
	const int blk_size_c = c%gridDim.x ? c/gridDim.x+1 : c/gridDim.x;
	while(c_offset<blk_size_n*blk_size_c){
		int offset_x = XOFFSET(c_offset + threadIdx.x);
		int offset_y = YOFFSET(c_offset + threadIdx.x);
		int x_pos = blockIdx.x * blk_size_c + XOFFSET(c_offset + threadIdx.x);
		int y_pos = blockIdx.y * blk_size_n + YOFFSET(c_offset + threadIdx.x);
		int idx1 = y_pos * c + x_pos;
		if(offset_x < blk_size_c && offset_y < blk_size_n){//WITHOUT THIS: THE C MUST BE MULTIPLE TIMES OF BLOCKDIM.X IN CURRENT IMPLEMENTATION !!!
			y[idx1] = x[idx1]*x[idx1];
		}
		c_offset += blockDim.x;
	}
	__syncthreads();

	//sum along block
	c_offset=0;
	Dtype res = 0;
	while(c_offset<blk_size_n*blk_size_c){
		int len = (c_offset + blockDim.x)<blk_size_n*blk_size_c ? blockDim.x : (blk_size_n*blk_size_c-c_offset);//valid threads to process
		while(len/2>0){
			if(threadIdx.x<len/2){

				int x_pos = blockIdx.x * blk_size_c + XOFFSET(c_offset + threadIdx.x);
				int y_pos = blockIdx.y * blk_size_n + YOFFSET(c_offset + threadIdx.x);
				int idx1 = y_pos * c + x_pos;
				x_pos = blockIdx.x * blk_size_c + XOFFSET(c_offset + threadIdx.x + (len+1)/2);
				y_pos = blockIdx.y * blk_size_n + YOFFSET(c_offset + threadIdx.x + (len+1)/2);
				int idx2 = y_pos * c + x_pos;
				//BUG: we must ALWAYS store this data. Use shared memory with size of blk_size_n*blk_size_c!!!
				y[idx1] += y[idx2];
			}
			__syncthreads();
			len=(len+1)/2;
		}

		int x_pos = blockIdx.x * blk_size_c + XOFFSET(c_offset);
		int y_pos = blockIdx.y * blk_size_n + YOFFSET(c_offset);
		int idx1 = y_pos * c + x_pos;
		res += y[idx1];
		c_offset += blockDim.x;
	}
	__syncthreads();

	//copy
	c_offset=0;
	while(c_offset<blk_size_n*blk_size_c){
		int offset_x = XOFFSET(c_offset + threadIdx.x);
		int offset_y = YOFFSET(c_offset + threadIdx.x);
		int x_pos = blockIdx.x * blk_size_c + XOFFSET(c_offset + threadIdx.x);
		int y_pos = blockIdx.y * blk_size_n + YOFFSET(c_offset + threadIdx.x);
		int idx1 = y_pos * c + x_pos;
		if(offset_x < blk_size_c && offset_y < blk_size_n){
			if(res){
				y[idx1] = Dtype(sqrt(res));
			}else{
				y[idx1] = Dtype(0);
			}
		}
	  	c_offset += blockDim.x;
	}
}

//Usage: dim3 block(c,1); dim3 thread(1,n); col_group_length_kernel<<<block,thread>>>(n,c,x,y,z);
template  <typename Dtype>
__global__ void col_group_length_kernel(const int n, const int c, const Dtype *x, Dtype* y, Dtype* z){
	int n_offset = 0;
	//initialize y
	while(n_offset<n){
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
				int idx1 = (n_offset+threadIdx.y)*c+blockIdx.x;
				int idx2 = (n_offset+threadIdx.y+(len+1)/2)*c+blockIdx.x;
				y[idx1] += y[idx2];
			}
			__syncthreads();
			len=(len+1)/2;
		}

		res += y[n_offset*c+blockIdx.x];
		n_offset += blockDim.y;
	}
	__syncthreads();

	if(res>0 && 0==threadIdx.y){
		z[blockIdx.x] = Dtype(sqrt(res));
	}else if(0==threadIdx.y) {
		z[blockIdx.x] = Dtype(0);
	}

	//copy
	n_offset=0;
	while(n_offset<n){
		int idx1 = (n_offset+threadIdx.y)*c + blockIdx.x;
		if(n_offset+threadIdx.y < n){
			if(res>0){
				y[idx1] = Dtype(sqrt(res));
			}else{
				y[idx1] = Dtype(0);
			}
		}
		n_offset += blockDim.y;
	}
}

//Usage: dim3 block(1,n); dim3 thread(c,1); row_group_length_kernel<<<block,thread>>>(n,c,x,y,z);
template  <typename Dtype>
__global__ void row_group_length_kernel(const int n, const int c, const Dtype *x, Dtype* y, Dtype* z){
	int c_offset = 0;
	//initialize y
	while(c_offset<c){
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
				int idx1 = blockIdx.y * c + c_offset + threadIdx.x;
				int idx2 = blockIdx.y * c + c_offset + threadIdx.x + (len+1)/2;
				y[idx1] += y[idx2];
			}
			__syncthreads();
			len=(len+1)/2;
		}

		res += y[blockIdx.y * c + c_offset];
		c_offset += blockDim.x;
	}
	__syncthreads();

	if(res>0 && 0==threadIdx.x){
		z[blockIdx.y] = Dtype(sqrt(res));
	}else if(0==threadIdx.x) {
		z[blockIdx.y] = Dtype(0);
	}

	//copy
	c_offset=0;
	while(c_offset<c){
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
/*
template <>
void caffe_gpu_bar_group_length<int>(const int n, const int c, const int* x, int* y, int* z, bool along_column_or_row){
	NOT_IMPLEMENTED;
}

template <>
void caffe_gpu_bar_group_length<unsigned int>(const int n, const int c, const unsigned int* x, unsigned int* y, unsigned int* z, bool along_column_or_row){
	NOT_IMPLEMENTED;
}
*/
template <>
void caffe_gpu_bar_group_length<float>(const int n, const int c, const float* x, float* y, float* z, bool along_column_or_row){
	int threads_per_block = Caffe::get_threads_per_block();
	if(along_column_or_row){
		dim3 block(c,1);
		dim3 thread(1,n>threads_per_block ? threads_per_block:n );//CAFFE_CUDA_NUM_THREADS
		col_group_length_kernel<<<block,thread>>>(n,c,x,y,z);
	}else{
		dim3 block(1,n);
		dim3 thread(c>threads_per_block ? threads_per_block:c, 1);//CAFFE_CUDA_NUM_THREADS
		row_group_length_kernel<<<block,thread>>>(n,c,x,y,z);
	}
	CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_bar_group_length<double>(const int n, const int c, const double* x, double* y, double* z, bool along_column_or_row){
	int threads_per_block = Caffe::get_threads_per_block();
	if(along_column_or_row){
		dim3 block(c,1);
		dim3 thread(1,n>threads_per_block ? threads_per_block:n );//CAFFE_CUDA_NUM_THREADS
		col_group_length_kernel<<<block,thread>>>(n,c,x,y,z);
	}else{
		dim3 block(1,n);
		dim3 thread(c>threads_per_block ? threads_per_block:c, 1);//CAFFE_CUDA_NUM_THREADS
		row_group_length_kernel<<<block,thread>>>(n,c,x,y,z);
	}
	CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_block_length<float>(const int n, const int c,
		const int blk_size_n, const int blk_size_c,
		const float *x, float* y){
	CHECK_LE(blk_size_n,n);
	CHECK_LE(blk_size_c,c);
	CHECK_EQ(n%blk_size_n,0);
	CHECK_EQ(c%blk_size_c,0);
	int threads_per_block = Caffe::get_threads_per_block();
	const int blk_num_n = (n+blk_size_n-1)/blk_size_n;
	const int blk_num_c = (c+blk_size_c-1)/blk_size_c;
	const int blk_size = blk_size_n*blk_size_c;
	dim3 block(blk_num_c,blk_num_n);
	dim3 thread(blk_size>threads_per_block?threads_per_block:blk_size, 1);
	block_length_kernel<<<block,thread>>>(n, c,x,y);
	CUDA_POST_KERNEL_CHECK;
}

template <>
void caffe_gpu_block_length<double>(const int n, const int c,
		const int blk_size_n, const int blk_size_c,
		const double *x, double* y){
	NOT_IMPLEMENTED;
}
template <>
void caffe_gpu_block_length<int>(const int n, const int c,
		const int blk_size_n, const int blk_size_c,
		const int *x, int* y){
	NOT_IMPLEMENTED;
}
template <>
void caffe_gpu_block_length<unsigned int>(const int n, const int c,
		const int blk_size_n, const int blk_size_c,
		const unsigned int *x, unsigned int* y){
	NOT_IMPLEMENTED;
}

}  // namespace caffe
