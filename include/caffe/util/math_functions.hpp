#ifndef CAFFE_UTIL_MATH_FUNCTIONS_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

namespace caffe {

// Caffe gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
template <typename Dtype>
void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

template <typename Dtype>
void caffe_cpu_cblas_gemm(const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const int lda, const Dtype* B, const int ldb, const Dtype beta,
    Dtype* C, const int ldc);

template <typename Dtype>
void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);

template <typename Dtype>
void caffe_axpy(const int N, const Dtype alpha, const Dtype* X,
    Dtype* Y);

template <typename Dtype>
void caffe_cpu_axpby(const int N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y);

template <typename Dtype>
void caffe_copy(const int N, const Dtype *X, Dtype *Y);

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype *X);

inline void caffe_memset(const size_t N, const int alpha, void* X) {
  memset(X, alpha, N);  // NOLINT(caffe/alt_fn)
}

template <typename Dtype>
void caffe_add_scalar(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_scal(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_sqr(const int N, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_div_checkzero(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);

template <typename Dtype>
void caffe_powx_seperate(const int n, const Dtype* a, const Dtype b, Dtype* y);

unsigned int caffe_rng_rand();

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype mu, const Dtype sigma,
                        Dtype* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r);

template <typename Dtype>
void caffe_exp(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_log(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_abs(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y);

template <typename Dtype>
Dtype caffe_cpu_strided_dot(const int n, const Dtype* x, const int incx,
    const Dtype* y, const int incy);

// sparse matrix A *  dense matrix B
// A is stored in CSR format
template <typename Dtype>
void caffe_cpu_sparse_mmcsr(const int M, const int N, const int K,
    const Dtype alpha,
    const Dtype* A_nonzero_buf, const int* A_nonzero_idx_buf, const int* A_idx_pointerB_,const int* A_idx_pointerE_,
    const Dtype* B,
    const Dtype beta,Dtype* C);

// dense matrix A to sparse matrix A in CSR format
template <typename Dtype>
void caffe_cpu_sparse_dense2csr(const int M, const int N,
    Dtype* A,
    Dtype* A_nonzero_buf, int* A_nonzero_idx_buf, int* A_idx_pointer_buf);

// Returns the sum of the absolute values of the elements of vector x
template <typename Dtype>
Dtype caffe_cpu_asum(const int n, const Dtype* x);

// Returns the column(true)/row(false) sums of the absolute values of the elements of matrix X
template <typename Dtype>
void caffe_cpu_asum_along_col_row(const int M, const int N, const Dtype* X, Dtype* y, bool dimen = true);

// the branchless, type-safe version from
// http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template<typename Dtype>
inline int8_t caffe_sign(Dtype val) {
  return (Dtype(0) < val) - (val < Dtype(0));
}
template<typename Dtype>
inline int8_t caffe_if_zerout(Dtype val) {
	Dtype thre = Dtype(ZEROUT_THRESHOLD);
	if(val<thre && val>(-thre)) return 1;
	else return 0;
}
template<typename Dtype>
inline int8_t caffe_if_nonzerout(Dtype val) {
	Dtype thre = Dtype(ZEROUT_THRESHOLD);
	if(val>=thre || val<=(-thre)) return 1;
	else return 0;
}
// The following two macros are modifications of DEFINE_VSL_UNARY_FUNC
//   in include/caffe/util/mkl_alternate.hpp authored by @Rowland Depp.
// Please refer to commit 7e8ef25c7 of the boost-eigen branch.
// Git cherry picking that commit caused a conflict hard to resolve and
//   copying that file in convenient for code reviewing.
// So they have to be pasted here temporarily.
#define DEFINE_CAFFE_CPU_UNARY_FUNC(name, operation) \
  template<typename Dtype> \
  void caffe_cpu_##name(const int n, const Dtype* x, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(x); CHECK(y); \
    for (int i = 0; i < n; ++i) { \
      operation; \
    } \
  }

// output is 1 for the positives, 0 for zero, and -1 for the negatives
DEFINE_CAFFE_CPU_UNARY_FUNC(sign, y[i] = caffe_sign<Dtype>(x[i]));
DEFINE_CAFFE_CPU_UNARY_FUNC(if_zerout, y[i] = caffe_if_zerout<Dtype>(x[i]));
DEFINE_CAFFE_CPU_UNARY_FUNC(if_nonzerout, y[i] = caffe_if_nonzerout<Dtype>(x[i]));
DEFINE_CAFFE_CPU_UNARY_FUNC(eltwise_multi, y[i] = y[i]*x[i]);

// This returns a nonzero value if the input has its sign bit set.
// The name sngbit is meant to avoid conflicts with std::signbit in the macro.
// The extra parens are needed because CUDA < 6.5 defines signbit as a macro,
// and we don't want that to expand here when CUDA headers are also included.
DEFINE_CAFFE_CPU_UNARY_FUNC(sgnbit, \
    y[i] = static_cast<bool>((std::signbit)(x[i])));

DEFINE_CAFFE_CPU_UNARY_FUNC(fabs, y[i] = std::fabs(x[i]));

template <typename Dtype>
void caffe_cpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);

//get if columns(true)/rows(false) in matrix X are all zeros
template <typename Dtype>
void caffe_cpu_if_all_zero(const int M, const int N, const Dtype *X, int* y, bool dimen=true);

//get the mask(0) of all-zero columns and rows
template <typename Dtype>
void caffe_cpu_all_zero_mask(const int M, const int N, const Dtype *X, Dtype* y);

//get column(true)/row(false) sparsity in matrix
template <typename Dtype>
Dtype caffe_cpu_group_sparsity(const int M, const int N, const Dtype *X, bool dimen=true);

//get masked cols
template <typename Dtype>
void caffe_cpu_del_zero_cols(const int M, const int N, const Dtype *x, Dtype *y, int * left_cols, const int* mask);

//remove all zero rows and columns, and concatenate remaining ones together
template <typename Dtype>
void caffe_cpu_concatenate_rows_cols(const int M, const int N, const Dtype *x, Dtype *y, const int* col_mask, const int* row_mask);

//dispatch dense rows in x to scattered rows in itself according to row_mask, assuming x is MxN dimension
template <typename Dtype>
void caffe_cpu_dispatch_rows(const int M, const int N, Dtype *x, const int* row_mask);

//get sqrt sum of weights within blocks and copy them at each position
template <typename Dtype>
void caffe_cpu_block_group_lasso(const int n, const int c,
		const int blk_size_n, const int blk_size_c,
		const Dtype *x, Dtype* y);

#ifndef CPU_ONLY  // GPU

// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.
template <typename Dtype>
void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

// dense matrix A *  sparse matrix B
// B is stored in CSR format
template <typename Dtype>
void caffe_gpu_sparse_mmcsr(const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A,
    const int nnz, const Dtype* B_nonzero_buf, const int* B_idx_pointer_buf, const int* B_nonzero_idx_buf,
    const Dtype beta,Dtype* C);

// sparse matrix A *  dense matrix B
// A is stored in CSR format
// transpose_C is required for temporary storage because of the column-major order of cusparse
template <typename Dtype>
void caffe_gpu_sparse_csrmm(const int M, const int N, const int K,
    const Dtype alpha,
    const int nnz, const Dtype* A_nonzero_buf, const int* A_idx_pointer_buf, const int* A_nonzero_idx_buf,
    const Dtype* B,
    const Dtype beta,
    Dtype* C, Dtype *transpose_C);

// dense matrix A to sparse matrix A in CSR format
template <typename Dtype>
void caffe_gpu_sparse_dense2csr(const int M, const int N,
    const Dtype* A, int* nnzPerRow,
    Dtype* A_nonzero_buf, int* A_idx_pointer_buf, int* A_nonzero_idx_buf,int *nnz_total);

template <typename Dtype>
void caffe_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);

template <typename Dtype>
void caffe_gpu_axpy(const int N, const Dtype alpha, const Dtype* X,
    Dtype* Y);

template <typename Dtype>
void caffe_gpu_zerout(void * mutable_gpu_data, int count, Dtype th);

template <typename Dtype>
void caffe_gpu_axpby(const int N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y);

void caffe_gpu_memcpy(const size_t N, const void *X, void *Y);

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype *X);

inline void caffe_gpu_memset(const size_t N, const int alpha, void* X) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaMemset(X, alpha, N));  // NOLINT(caffe/alt_fn)
#else
  NO_GPU;
#endif
}

template <typename Dtype>
void caffe_gpu_add_scalar(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_gpu_scal(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_gpu_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_div_checkzero(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_abs(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_gpu_exp(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_gpu_log(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_gpu_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);

// caffe_gpu_rng_uniform with two arguments generates integers in the range
// [0, UINT_MAX].
void caffe_gpu_rng_uniform(const int n, unsigned int* r);

// caffe_gpu_rng_uniform with four arguments generates floats in the range
// (a, b] (strictly greater than a, less than or equal to b) due to the
// specification of curandGenerateUniform.  With a = 0, b = 1, just calls
// curandGenerateUniform; with other limits will shift and scale the outputs
// appropriately after calling curandGenerateUniform.
template <typename Dtype>
void caffe_gpu_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r);

template <typename Dtype>
void caffe_gpu_rng_gaussian(const int n, const Dtype mu, const Dtype sigma,
                            Dtype* r);

template <typename Dtype>
void caffe_gpu_rng_bernoulli(const int n, const Dtype p, int* r);

template <typename Dtype>
void caffe_gpu_dot(const int n, const Dtype* x, const Dtype* y, Dtype* out);

template <typename Dtype>
void caffe_gpu_asum(const int n, const Dtype* x, Dtype* y, int stride = 1);

template<typename Dtype>
void caffe_gpu_sign(const int n, const Dtype* x, Dtype* y);

template<typename Dtype>
void caffe_gpu_if_zerout(const int n, const Dtype* x, Dtype* y);

template<typename Dtype>
void caffe_gpu_if_nonzerout(const int n, const Dtype* x, Dtype* y);

template<typename Dtype>
void caffe_gpu_eltwise_multi(const int n, const Dtype* x, Dtype* y);

template<typename Dtype>
void caffe_gpu_sgnbit(const int n, const Dtype* x, Dtype* y);

template <typename Dtype>
void caffe_gpu_fabs(const int n, const Dtype* x, Dtype* y);

template <typename Dtype>
void caffe_gpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);

//get sqrt sum of weights within bars(column(true)/row(false)) and copy them at each position
template <typename Dtype>
void caffe_gpu_bar_group_lasso(const int n, const int c, const Dtype *x, Dtype* y, bool along_column_or_row = true);

//get sqrt sum of weights within blocks and copy them at each position
template <typename Dtype>
void caffe_gpu_block_group_lasso(const int n, const int c,
		const int blk_size_n, const int blk_size_c,
		const Dtype *x, Dtype* y);

#define DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(name, operation) \
template<typename Dtype> \
__global__ void name##_kernel(const int n, const Dtype* x, Dtype* y) { \
  CUDA_KERNEL_LOOP(index, n) { \
    operation; \
  } \
} \
template <> \
void caffe_gpu_##name<int>(const int n, const int* x, int* y) { \
	NOT_IMPLEMENTED; \
} \
template <> \
void caffe_gpu_##name<unsigned int>(const int n, const unsigned int* x, unsigned int* y) { \
	NOT_IMPLEMENTED; \
} \
template <> \
void caffe_gpu_##name<float>(const int n, const float* x, float* y) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>( \
      n, x, y); \
} \
template <> \
void caffe_gpu_##name<double>(const int n, const double* x, double* y) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>( \
      n, x, y); \
}

#endif  // !CPU_ONLY

}  // namespace caffe

#endif  // CAFFE_UTIL_MATH_FUNCTIONS_H_
