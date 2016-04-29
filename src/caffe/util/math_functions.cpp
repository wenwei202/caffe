#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>
#include <omp.h>
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
  //LOG(INFO)<<"A("<<M<<"x"<<K<<")*B("<<K<<"x"<<N<<")=C("<<M<<"x"<<N<<") "<<M*K*N<<" Multi.";
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <>
void caffe_cpu_cblas_gemm<float>(const int M, const int N, const int K,
    const float alpha, const float* A, const int lda, const float* B, const int ldb, const float beta,
    float* C, const int ldc){
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, lda, B,
	      ldb, beta, C, ldc);
}

template <>
void caffe_cpu_cblas_gemm<double>(const int M, const int N, const int K,
    const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta,
    double* C, const int ldc){
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, lda, B,
	      ldb, beta, C, ldc);
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<unsigned int>(const int N, const unsigned int alpha, unsigned int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
      NO_GPU;
#endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}

template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

template <>
void caffe_div_checkzero<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDivCheckZero(n, a, b, y);
}

template <>
void caffe_div_checkzero<double>(const int n, const double* a, const double* b,
		double* y) {
  vdDivCheckZero(n, a, b, y);
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx_seperate<float>(const int n, const float* a, const float b,
    float* y) {
  for(int i=0;i<n;i++){
	  y[i] = pow(a[i], b);
  }
}
template <>
void caffe_powx_seperate<double>(const int n, const double* a, const double b,
		double* y) {
   for(int i=0;i<n;i++){
		  y[i] = pow(a[i], b);
   }
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}

template <>
void caffe_log<float>(const int n, const float* a, float* y) {
  vsLn(n, a, y);
}

template <>
void caffe_log<double>(const int n, const double* a, double* y) {
  vdLn(n, a, y);
}

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
    vdAbs(n, a, y);
}

unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_uniform<float>(const int n, const float a, const float b,
                              float* r);

template
void caffe_rng_uniform<double>(const int n, const double a, const double b,
                               double* r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype a,
                        const Dtype sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<Dtype> random_distribution(a, sigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_gaussian<float>(const int n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double>(const int n, const double mu,
                                const double sigma, double* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

template
float caffe_cpu_dot<float>(const int n, const float* x, const float* y);

template
double caffe_cpu_dot<double>(const int n, const double* x, const double* y);

template <>
void caffe_cpu_sparse_dense2csr<float>(const int M, const int N,
    float* A,
    float* A_nonzero_buf, int* A_nonzero_idx_buf, int* A_idx_pointer_buf){
#ifdef USE_MKL
	MKL_INT info;
	const MKL_INT job[] = {0,0,0,2,M*N,1};
	mkl_sdnscsr(job, &M , &N , A,
			&N , A_nonzero_buf, A_nonzero_idx_buf, A_idx_pointer_buf,  &info);
	if(info){
		LOG(FATAL)<<"The routine is interrupted processing the "<<
				info<<"-th row "
				<<"because there is no space in the arrays acsr and ja according to the value nzmax.";
	}
#else
	NOT_IMPLEMENTED;
#endif
}

template <>
void caffe_cpu_sparse_dense2csr<double>(const int M, const int N,
    double* A,
    double* A_nonzero_buf, int* A_nonzero_idx_buf, int* A_idx_pointer_buf){
#ifdef USE_MKL
	MKL_INT info;
	const MKL_INT job[] = {0,0,0,2,M*N,1};
	mkl_ddnscsr(job, &M , &N , A,
			&N , A_nonzero_buf, A_nonzero_idx_buf, A_idx_pointer_buf,  &info);
	if(info){
		LOG(FATAL)<<"The routine is interrupted processing the "<<
				info<<"-th row "
				<<"because there is no space in the arrays acsr and ja according to the value nzmax.";
	}
#else
	NOT_IMPLEMENTED;
#endif
}

template <>
void caffe_cpu_sparse_mmcsr<float>(const int M, const int N, const int K,
    const float alpha,
    const float* A_nonzero_buf, const int* A_nonzero_idx_buf, const int* A_idx_pointerB_,const int* A_idx_pointerE_,
    const float* B,
    const float beta,float* C){
#ifdef USE_MKL
	const char *matdescra = "GXXCX";//6 bytes
	const char transa = 'N';
	mkl_scsrmm (&transa, &M , &N, &K,
			&alpha , matdescra,
			A_nonzero_buf, A_nonzero_idx_buf, A_idx_pointerB_, A_idx_pointerE_,
			B, &N,
			&beta , C, &N);
//#pragma omp parallel
//	{
//	  const int BLOCK = 64;
//
//	  int nthreads = omp_get_num_threads();
//	  int tid = omp_get_thread_num();
//
//	  int total_work = A_idx_pointerB_[M];
//	  int work_per_thread = (total_work + nthreads - 1)/nthreads;
//
//	  int begin = tid == 0 ? 0 : std::lower_bound(A_idx_pointerB_, A_idx_pointerB_ + M, work_per_thread*tid) - A_idx_pointerB_;
//	  int end = tid == nthreads - 1 ? M : std::lower_bound(A_idx_pointerB_, A_idx_pointerB_ + M, work_per_thread*(tid + 1)) - A_idx_pointerB_;
//
//	  float sum[BLOCK];
//
//	  for (int b = 0; b < N/BLOCK; ++b) {
//		for (int i = begin; i < end; ++i) {
//		  for (int k = 0; k < BLOCK; ++k) {
//			sum[k] = 0;
//		  }
//		  for (int j = A_idx_pointerB_[i]; j < A_idx_pointerB_[i + 1]; ++j) {
//			float v = A_nonzero_buf[j];
//			int c = A_nonzero_idx_buf[j];
//
//			for (int k = 0; k < BLOCK; ++k) {
//			  sum[k] += v*B[c*N + k + b*BLOCK];
//			}
//		  }
//		  for (int k = 0; k < BLOCK; ++k) {
//			C[i*N + k + b*BLOCK] = sum[k];
//		  }
//		}
//	  }
//
//	  int rem = N - N/BLOCK*BLOCK;
//	  for (int i = begin; i < end; ++i) {
//		for (int k = 0; k < rem; ++k) {
//		  sum[k] = 0;
//		}
//		for (int j = A_idx_pointerB_[i]; j < A_idx_pointerB_[i + 1]; ++j) {
//		  float v = A_nonzero_buf[j];
//		  int c = A_nonzero_idx_buf[j];
//
//		  for (int k = 0; k < rem; ++k) {
//			sum[k] += v*B[c*N + k + (N/BLOCK)*BLOCK];
//		  }
//		}
//		for (int k = 0; k < rem; ++k) {
//		  C[i*N + k + (N/BLOCK)*BLOCK] = sum[k];
//		}
//	  }
//	}
#else
	NOT_IMPLEMENTED;
#endif
}

template <>
void caffe_cpu_sparse_mmcsr<double>(const int M, const int N, const int K,
    const double alpha,
    const double* A_nonzero_buf, const int* A_nonzero_idx_buf, const int* A_idx_pointerB_,const int* A_idx_pointerE_,
    const double* B,
    const double beta,double* C){
#ifdef USE_MKL
	char matdescra[6];
	matdescra[0] = 'g';
	matdescra[3] = 'c';
	const char transa = 'N';
	mkl_dcsrmm (&transa, &M , &N, &K,
			&alpha , matdescra,
			A_nonzero_buf, A_nonzero_idx_buf, A_idx_pointerB_, A_idx_pointerE_,
			B, &N,
			&beta , C, &N);
#else
	NOT_IMPLEMENTED;
#endif
}

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double caffe_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <>
int caffe_cpu_asum<int>(const int n, const int* x) {
  int sum = 0;
  for (int i=0;i<n;i++){
	  sum += abs(x[i]);
  }
  return sum;
}

template <>
void caffe_cpu_asum_along_col_row<float>(const int M, const int N, const float* X, float* y, bool dimen){
	if(dimen){//along column
		for(int i=0;i<N;i++){
			y[i] = cblas_sasum(M, X+i, N);
		}
	}else{//along row
		for(int i=0;i<M;i++){
			y[i] = cblas_sasum(N, X+i*N, 1);
		}
	}
}

template <>
void caffe_cpu_asum_along_col_row<double>(const int M, const int N, const double* X, double* y, bool dimen){
	if(dimen){//along column
		for(int i=0;i<N;i++){
			y[i] = cblas_dasum(M, X+i, N);
		}
	}else{//along row
		for(int i=0;i<M;i++){
			y[i] = cblas_dasum(N, X+i*N, 1);
		}
	}
}

template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

template <typename Dtype>
void caffe_cpu_if_all_zero(const int M, const int N, const Dtype *x, int* y, bool dimen){
	if(dimen){//along columns
		for(int col=0; col<N; ++col){
			y[col]=true;
			for(int row=0; row<M; row++){
				if(x[col+row*N]!=0){
					y[col] = false;
					break;
				}
			}
		}
	}else{//along rows
		for(int row=0; row<M; ++row){
			y[row]=true;
			for(int col=0; col<N; col++){
				if(x[col+row*N]!=0){
					y[row] = false;
					break;
				}
			}
		}
	}
}
template
void caffe_cpu_if_all_zero(const int M, const int N, const float *x, int* y, bool dimen);
template
void caffe_cpu_if_all_zero(const int M, const int N, const double *x, int* y, bool dimen);

template <typename Dtype>
void caffe_cpu_all_zero_mask(const int M, const int N, const Dtype *X, Dtype* Y){
	//along rows
	Dtype val = (Dtype)1;
	for(int row=0; row<M; ++row){
		val = (Dtype)0;
		for(int col=0; col<N; col++){
			if(X[col+row*N]!=0){
				val = (Dtype)1;
				break;
			}
		}
		caffe_set(N,val,Y+row*N);
	}
	//along columns
	for(int col=0; col<N; ++col){
		val = (Dtype)0;
		for(int row=0; row<M; row++){
			if(X[col+row*N]!=0){
				val = (Dtype)1;
				break;
			}
		}
		if(!val){//only set 0
			for(int row=0; row<M; row++){
				Y[col+row*N] = val;
			}
		}
	}
}
template
void caffe_cpu_all_zero_mask(const int M, const int N, const float *X, float* y);
template
void caffe_cpu_all_zero_mask(const int M, const int N, const double *X, double* y);
template
void caffe_cpu_all_zero_mask(const int M, const int N, const int *X, int* y);
template
void caffe_cpu_all_zero_mask(const int M, const int N, const unsigned int *X, unsigned int* y);

template <typename Dtype>
Dtype caffe_cpu_group_sparsity(const int M, const int N, const Dtype *x, bool dimen){
	Dtype sparsity = (Dtype)0;
	int counter = 0;
	if(dimen){//along columns
		for(int col=0; col<N; ++col){
			counter++;
			for(int row=0; row<M; row++){
				if(x[col+row*N]!=0){
					counter--;
					break;
				}
			}
		}
		sparsity = (Dtype)counter/(Dtype)N;
	}else{//along rows
		for(int row=0; row<M; ++row){
			counter++;
			for(int col=0; col<N; col++){
				if(x[col+row*N]!=0){
					counter--;
					break;
				}
			}
		}
		sparsity = (Dtype)counter/(Dtype)M;
	}
	return sparsity;
}
template float caffe_cpu_group_sparsity(const int M, const int N, const float *x, bool dimen);
template double caffe_cpu_group_sparsity(const int M, const int N, const double *x, bool dimen);

template <typename Dtype>
void caffe_cpu_del_zero_cols(const int M, const int N, const Dtype *x, Dtype *y, int * left_cols, const int* mask){
	int dst_col = 0;
	for(int row=0; row<M; row++){
		dst_col = 0;
		for(int src_col=0; src_col<N; src_col++){
			if(!mask[src_col]){
				//if(src_col!=dst_col){
				//	x[row*N+dst_col] = x[row*N+src_col];
				//}
				y[row*N+dst_col] = x[row*N+src_col];
				dst_col++;
			}
		}
	}
	*left_cols = dst_col;
}
template
void  caffe_cpu_del_zero_cols<float>(const int M, const int N, const float *x, float *y, int * left_cols, const int* mask);
template
void  caffe_cpu_del_zero_cols<double>(const int M, const int N, const double *x, double *y, int * left_cols, const int* mask);

template <typename Dtype>
void caffe_cpu_concatenate_rows_cols(const int M, const int N, const Dtype *x, Dtype *y, const int* col_mask, const int* row_mask){
	int left_cols = 0;
	for(int i=0;i<N;i++){
		left_cols += !col_mask[i];
	}

	int cur_row = 0;
	for(int row=0;row<M;row++){
		if(!row_mask[row]){
			int cur_col = 0;
			for(int col=0;col<N;col++){
				if(!col_mask[col]){
					y[cur_row*left_cols+cur_col] = x[row*N+col];
					cur_col++;
				}
			}
			CHECK_EQ(cur_col,left_cols);
			cur_row++;
		}
	}
}
template void caffe_cpu_concatenate_rows_cols<float>(const int M, const int N, const float *x, float *y, const int* col_mask, const int* row_mask);
template void caffe_cpu_concatenate_rows_cols<double>(const int M, const int N, const double *x, double *y, const int* col_mask, const int* row_mask);

template <typename Dtype>
void caffe_cpu_dispatch_rows(const int M, const int N, Dtype *x, const int* row_mask){
	int total_nonzero_rows = 0;
	for(int row=0; row<M; row++){
		total_nonzero_rows += !row_mask[row];
	}

	int src_row = total_nonzero_rows-1;
	for(int row=M-1; row>=0; row--){
		if(!row_mask[row]){//nonzero rows
			CHECK_GE(src_row,0);
			CHECK_LE(src_row,row);
			if(src_row!=row) caffe_copy(N,x+src_row*N,x+row*N);
			src_row--;
		}else{//set to zeros
			caffe_set(N,(Dtype)0.0,x+row*N);
		}
	}
}
template void caffe_cpu_dispatch_rows<float>(const int M, const int N, float *x, const int* row_mask);
template void caffe_cpu_dispatch_rows<double>(const int M, const int N, double *x, const int* row_mask);

template <typename Dtype>
void caffe_cpu_block_group_lasso(const int n, const int c,
		const int blk_size_n, const int blk_size_c,
		const Dtype *x, Dtype* y){
	  CHECK_LE(blk_size_n,n);
	  CHECK_LE(blk_size_c,c);
	  CHECK_EQ(n%blk_size_n,0);
	  CHECK_EQ(c%blk_size_c,0);
	  int block_num_c = c/blk_size_c;
	  int block_num_n = n/blk_size_n;
	  Dtype sum_val = 0;
	  for(int bn=0;bn<block_num_n;bn++){
		  for(int bc=0;bc<block_num_c;bc++){
			  sum_val = 0;
			  for(int n_idx=0;n_idx<blk_size_n;n_idx++){
			  	  for(int c_idx=0;c_idx<blk_size_c;c_idx++){
			  		  int idx = (bn*blk_size_n+n_idx)*c + (bc*blk_size_c+c_idx);
			  		  sum_val += x[idx]*x[idx];
			      }
			  }
			  for(int n_idx=0;n_idx<blk_size_n;n_idx++){
			  	  for(int c_idx=0;c_idx<blk_size_c;c_idx++){
			  		  int idx = (bn*blk_size_n+n_idx)*c + (bc*blk_size_c+c_idx);
			  		  if(sum_val>0) y[idx] = sqrt(sum_val);
			  		  else y[idx] = 0;
			      }
			  }
		  }
	  }
}
template void  caffe_cpu_block_group_lasso<float>(const int n, const int c,
		const int blk_size_n, const int blk_size_c,
		const float *x, float* y);
template void  caffe_cpu_block_group_lasso<double>(const int n, const int c,
		const int blk_size_n, const int blk_size_c,
		const double *x, double* y);

}  // namespace caffe
