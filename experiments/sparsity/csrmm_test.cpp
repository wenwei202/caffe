#include <stdlib.h>
#include <string.h>

#include <vector>

#include <omp.h>
#include <mkl.h>
#include <immintrin.h>

#include "SpMP/CSR.hpp"
#include "SpMP/reordering/BFSBipartite.hpp"
#include "SpMP/test/test.hpp"
#include "SpMP/synk/barrier.hpp"

#ifdef USE_LIBXSMM
#ifdef MKL_DIRECT_CALL
#undef MKL_DIRECT_CALL
#include "libxsmm.h"
#define MKL_DIRECT_CALL
#else
#include "libxsmm.h"
#endif
#endif

using namespace std;
using namespace SpMP;

static void printEfficiency(
  double *times, int REPEAT, double flop, double denseFlop, double byte, double loweringTime)
{
  sort(times, times + REPEAT);

  double t = times[REPEAT/2];

  //printf(
    //"%g sec %7.2f sparse_gflops %7.2f dense_gflops %7.2f sparse_gbps %7.2f dense_gflops(including_lowering_overhead)\n",
    //t, flop/t/1e9, denseFlop/t/1e9, byte/t/1e9, denseFlop/(t + loweringTime)/1e9);
  printf(
    "%7.2f dense_gflops %7.2f dense_gflops(including_lowering_overhead)\n",
    denseFlop/t/1e9, denseFlop/(t + loweringTime)/1e9);
}

/**
 * @param A M*K CSR matrix
 * @param B K*N row-major dense matrix
 * @param C M*N row-major dense matrix
 */
void my_scsrmm(
  int M, int N, int K,
  const float *A_values, const int *A_colidx, const int *A_rowptr, 
  const float *B, int ldaB,
  float *C, int ldaC, bool serial = false)
{
//#define BLOCK (128)

  int begin, end;

  if (serial) {
    begin = 0;
    end = M;
  }
  else {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    int total_work = A_rowptr[M];
    int work_per_thread = (total_work + nthreads - 1)/nthreads;

    begin = tid == 0 ? 0 : std::lower_bound(A_rowptr, A_rowptr + M, work_per_thread*tid) - A_rowptr;
    end = tid == nthreads - 1 ? M : std::lower_bound(A_rowptr, A_rowptr + M, work_per_thread*(tid + 1)) - A_rowptr;
  }

#ifdef BLOCK
  float sum[BLOCK];

  for (int b = 0; b < N/BLOCK; ++b) {
    for (int i = begin; i < end; ++i) {
      for (int k = 0; k < BLOCK; ++k) {
        sum[k] = 0;
      }
      for (int j = A_rowptr[i]; j < A_rowptr[i + 1]; ++j) {
        float v = A_values[j];
        int c = A_colidx[j];

//#pragma simd
        for (int k = 0; k < BLOCK; ++k) {
          sum[k] += v*B[c*N + k + b*BLOCK];
        }
      }
//#pragma vector nontemporal(C)
      for (int k = 0; k < BLOCK; ++k) {
        C[i*N + k + b*BLOCK] = sum[k];
      }
    }
  }

  int rem = N - N/BLOCK*BLOCK;
  for (int i = begin; i < end; ++i) {
    for (int k = 0; k < rem; ++k) {
      sum[k] = 0;
    }
    for (int j = A_rowptr[i]; j < A_rowptr[i + 1]; ++j) {
      float v = A_values[j];
      int c = A_colidx[j];

      for (int k = 0; k < rem; ++k) {
        sum[k] += v*B[c*N + k + (N/BLOCK)*BLOCK];
      }
    }
    for (int k = 0; k < rem; ++k) {
      C[i*N + k + (N/BLOCK)*BLOCK] = sum[k];
    }
  }
#else
  float sum[N];

  for (int i = begin; i < end; ++i) {
    for (int k = 0; k < N; ++k) {
      sum[k] = 0;
    }
    for (int j = A_rowptr[i]; j < A_rowptr[i + 1]; ++j) {
      float v = A_values[j];
      int c = A_colidx[j];

//#pragma simd
      for (int k = 0; k < N; ++k) {
        sum[k] += v*B[c*N + k];
        /*if (i*N + k == 5) {
          printf("%g*%g +", v, B[c*N + k]);
        }*/
      }
    }
    //if (i*N <= 5 && (i + 1)*N > 5) printf(" = %g, %d:%d\n", sum[5], A_rowptr[i], A_rowptr[i + 1]);
//#pragma vector nontemporal(C)
    for (int k = 0; k < N; ++k) {
      C[i*N + k] = sum[k];
    }
  }
#endif
}

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_col,int* all_zero_mask, int * feature_map_mask) {
  //get zero and nonzero maps
	int kernel_slice_dim = kernel_w*kernel_h;

  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  int forward_count = 0;
#pragma omp parallel for
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    //int sum=0;
    //for(int ii=0;ii<forwarding_mask.size();ii++){
    //	sum+=forwarding_mask[ii];
    //}
    //if(all_zero_mask && all_zero_mask[c]/*feature_map_mask && !feature_map_mask[c_im]*/) {
    	//continue;
    //}
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h + h_offset;
        int w_pad = w * stride_w + w_offset;
        //if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width){
          //data_col[(c * height_col + h) * width_col + w] =
        	data_col[(c * height_col + h) * width_col + w] =
            data_im[(c_im * (height + pad_h) + h_pad) * (width + pad_w) + w_pad];
        //}
        //else{
          //data_col[(c * height_col + h) * width_col + w] = 0;
        	//data_col[(forward_count * height_col + h) * width_col + w] = 0;
        //}
      }
    }
    //forward_count++;
  }
}

class KernelTensor
{
public :
  KernelTensor(const CSR *Ain, int nInChannels, int width, int pad) : width(width), pad(pad) {

    A = new CSR(Ain->m, nInChannels*(width + pad)*(width + pad), Ain->getNnz());

    k = sqrt(Ain->n/nInChannels);
    assert(nInChannels*k*k == Ain->n);

    // check sparsity of each (output_channel, input_channel)
    int **nnz_per_channel_pair = new int *[A->m];
    for (int i = 0; i < A->m; ++i) {
      nnz_per_channel_pair[i] = new int[nInChannels];
      memset(nnz_per_channel_pair[i], 0, sizeof(int)*nInChannels);
    }
    nNonZeroKernels = 0;
    for (int i = 0; i < Ain->m; ++i) {
      A->rowptr[i] = Ain->rowptr[i];
      for (int j = Ain->rowptr[i]; j < Ain->rowptr[i + 1]; ++j) {
        int c = Ain->colidx[j];
        int ic = c/(k*k);

        A->colidx[j] = (ic*(width + pad) + (c/k)%k)*(width + pad) + c%k;
        A->values[j] = Ain->values[j];

        nnz_per_channel_pair[i][ic]++;
      }

      for (int j = 0; j < nInChannels; ++j) {
        if (nnz_per_channel_pair[i][j] != 0) {
          ++nNonZeroKernels;
        }
      }
    }
    A->rowptr[A->m] = Ain->rowptr[A->m];

    for (int i = 0; i < A->m; ++i) {
      delete[] nnz_per_channel_pair[i];
    }
    delete[] nnz_per_channel_pair;

    int *rowPerm = new int[A->m], *rowInversePerm = new int[A->m];
    colPerm = new int[A->n], colInversePerm = new int[A->n];

    //CSR *AT = A->transpose();
    //bfsBipartite(*A, *AT, rowPerm, rowInversePerm, colPerm, colInversePerm);
    //FREE(A->diagptr);
    //int bw = A->getBandwidth();
    //double avgW = A->getAverageWidth();
    //A->permuteColsInPlace(colPerm);

    //printf("BW is reduced by BFS reordering: %d -> %d\n", bw, A->getBandwidth());
    //printf("Average width is reduced by BFS reordering: %g -> %g\n", avgW, A->getAverageWidth());

    //delete[] rowPerm;
    //delete[] rowInversePerm;

    //delete AT;

    posix_memalign((void **)&values, 4096, sizeof(float)*A->getNnz());
    for (int i = 0; i < A->getNnz(); ++i) {
      values[i] = A->values[i];
    }
    FREE(A->values);
  }

  ~KernelTensor() {
    delete A;
  }

  void conv(float *in, float *out, int stride, bool serial = false) {
    int begin, end;
    if (serial) {
      begin = 0;
      end = A->m;
    }
    else {
      int nthreads = omp_get_num_threads();
      int tid = omp_get_thread_num();

      int total_work = A->rowptr[A->m];
      int work_per_thread = (total_work + nthreads - 1)/nthreads;

      begin = tid == 0 ? 0 : std::lower_bound(A->rowptr, A->rowptr + A->m, work_per_thread*tid) - A->rowptr;
      end = tid == nthreads - 1 ? A->m : std::lower_bound(A->rowptr, A->rowptr + A->m, work_per_thread*(tid + 1)) - A->rowptr;
    }

    if (width == 13 && pad == 1 && stride == 1 && k == 3) {
      int WIDTH = 13;
      int WOUT = 13;
      int PAD = 1;
      
      __m256 sum[(WOUT + 1)/2][2];
      __declspec(aligned(64)) float sum_temp[8];

      for (int i = begin; i < end; ++i) {
        if (A->rowptr[i + 1] == A->rowptr[i]) continue;

        // Upper half of images
        int hbegin = 0, hend = (WOUT + 1)/2;
        int j = A->rowptr[i];
        __m256 c = _mm256_set1_ps(values[j]);
        int off = A->colidx[j];

        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_mul_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD)));
          sum[h - hbegin][1] = _mm256_mul_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 8));
        }

        int jbegin = A->rowptr[i] + 1;
        int jend = A->rowptr[i + 1];

        for (j = jbegin; j < jend; ++j) {
          c = _mm256_set1_ps(values[j]);
          off = A->colidx[j];

          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin][0] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
            sum[h - hbegin][1] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
          }
        }

        for (int h = hbegin; h < hend; ++h) {
          _mm256_store_ps(out + (i*WOUT + h)*WOUT, sum[h - hbegin][0]);
          _mm256_store_ps(sum_temp, sum[h - hbegin][1]);
          for (int w = 8; w < WOUT; ++w) {
            out[i*WOUT*WOUT + h*WOUT + w] = sum_temp[w - 8];
          }
        }

        // Lower half of images
        hbegin = (WOUT + 1)/2; hend = WOUT;
        j = A->rowptr[i];
        c = _mm256_set1_ps(values[j]);
        off = A->colidx[j];

        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_mul_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD)));
          sum[h - hbegin][1] = _mm256_mul_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 8));
        }

        for (j = jbegin; j < jend; ++j) {
          c = _mm256_set1_ps(values[j]);
          off = A->colidx[j];

          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin][0] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
            sum[h - hbegin][1] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
          }
        }

        for (int h = hbegin; h < hend; ++h) {
          _mm256_store_ps(out + (i*WOUT + h)*WOUT, sum[h - hbegin][0]);
          _mm256_store_ps(sum_temp, sum[h - hbegin][1]);
          for (int w = 8; w < WOUT; ++w) {
            out[i*WOUT*WOUT + h*WOUT + w] = sum_temp[w - 8];
          }
        }
      }

      return;
    }
    else if (width == 27 && pad == 2 && stride == 1 && k == 5) {
      //return conv_<27, 27, 2>(in, out, stride, serial);
      int WIDTH = 27;
      int WOUT = 27;
      int PAD = 2;
      
      __m256 sum[(WOUT + 1)/4][2];
      __declspec(aligned(64)) float sum_temp[8];

      for (int i = begin; i < end; ++i) {
        if (A->rowptr[i + 1] == A->rowptr[i]) continue;

        // (0, 0) block
        int hbegin = 0, hend = (WOUT + 1)/4;
        int j = A->rowptr[i];
        __m256 c = _mm256_set1_ps(values[j]);
        int off = A->colidx[j];

        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_mul_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD)));
          sum[h - hbegin][1] = _mm256_mul_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 8));
        }

        int jbegin = A->rowptr[i] + 1;
        int jend = A->rowptr[i + 1];

        for (j = jbegin; j < jend; ++j) {
          c = _mm256_set1_ps(values[j]);
          off = A->colidx[j];

          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin][0] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
            sum[h - hbegin][1] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
          }
        }

        for (int h = hbegin; h < hend; ++h) {
          _mm256_store_ps(out + (i*WOUT + h)*WOUT, sum[h - hbegin][0]);
          _mm256_store_ps(out + (i*WOUT + h)*WOUT + 8, sum[h - hbegin][1]);
        }

        // (0, 1) block
        j = A->rowptr[i];
        c = _mm256_set1_ps(values[j]);
        off = A->colidx[j];

        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_mul_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 16));
          sum[h - hbegin][1] = _mm256_mul_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 24));
        }

        for (j = jbegin; j < jend; ++j) {
          c = _mm256_set1_ps(values[j]);
          off = A->colidx[j];

          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin][0] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 16), sum[h - hbegin][0]);
            sum[h - hbegin][1] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 24), sum[h - hbegin][1]);
          }
        }

        for (int h = hbegin; h < hend; ++h) {
          _mm256_store_ps(out + (i*WOUT + h)*WOUT + 16, sum[h - hbegin][0]);
          _mm256_store_ps(sum_temp, sum[h - hbegin][1]);
          for (int w = 24; w < WOUT; ++w) {
            out[i*WOUT*WOUT + h*WOUT + w] = sum_temp[w - 24];
          }
        }

        // (1, 0) block
        hbegin = (WOUT + 1)/4; hend = (WOUT + 1)/4*2;

        j = A->rowptr[i];
        c = _mm256_set1_ps(values[j]);
        off = A->colidx[j];

        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_mul_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD)));
          sum[h - hbegin][1] = _mm256_mul_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 8));
        }

        for (j = jbegin; j < jend; ++j) {
          c = _mm256_set1_ps(values[j]);
          off = A->colidx[j];

          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin][0] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
            sum[h - hbegin][1] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
          }
        }

        for (int h = hbegin; h < hend; ++h) {
          _mm256_store_ps(out + (i*WOUT + h)*WOUT, sum[h - hbegin][0]);
          _mm256_store_ps(out + (i*WOUT + h)*WOUT + 8, sum[h - hbegin][1]);
        }

        // (1, 1) block
        j = A->rowptr[i];
        c = _mm256_set1_ps(values[j]);
        off = A->colidx[j];

        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_mul_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 16));
          sum[h - hbegin][1] = _mm256_mul_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 24));
        }

        for (j = jbegin; j < jend; ++j) {
          c = _mm256_set1_ps(values[j]);
          off = A->colidx[j];

          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin][0] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 16), sum[h - hbegin][0]);
            sum[h - hbegin][1] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 24), sum[h - hbegin][1]);
          }
        }

        for (int h = hbegin; h < hend; ++h) {
          _mm256_store_ps(out + (i*WOUT + h)*WOUT + 16, sum[h - hbegin][0]);
          _mm256_store_ps(sum_temp, sum[h - hbegin][1]);
          for (int w = 24; w < WOUT; ++w) {
            out[i*WOUT*WOUT + h*WOUT + w] = sum_temp[w - 24];
          }
        }

        // (2, 0) block
        hbegin = (WOUT + 1)/4*2; hend = (WOUT + 1)/4*3;

        j = A->rowptr[i];
        c = _mm256_set1_ps(values[j]);
        off = A->colidx[j];

        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_mul_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD)));
          sum[h - hbegin][1] = _mm256_mul_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 8));
        }

        for (j = jbegin; j < jend; ++j) {
          c = _mm256_set1_ps(values[j]);
          off = A->colidx[j];

          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin][0] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
            sum[h - hbegin][1] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
          }
        }

        for (int h = hbegin; h < hend; ++h) {
          _mm256_store_ps(out + (i*WOUT + h)*WOUT, sum[h - hbegin][0]);
          _mm256_store_ps(out + (i*WOUT + h)*WOUT + 8, sum[h - hbegin][1]);
        }

        // (2, 1) block
        j = A->rowptr[i];
        c = _mm256_set1_ps(values[j]);
        off = A->colidx[j];

        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_mul_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 16));
          sum[h - hbegin][1] = _mm256_mul_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 24));
        }

        for (j = jbegin; j < jend; ++j) {
          c = _mm256_set1_ps(values[j]);
          off = A->colidx[j];

          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin][0] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 16), sum[h - hbegin][0]);
            sum[h - hbegin][1] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 24), sum[h - hbegin][1]);
          }
        }

        for (int h = hbegin; h < hend; ++h) {
          _mm256_store_ps(out + (i*WOUT + h)*WOUT + 16, sum[h - hbegin][0]);
          _mm256_store_ps(sum_temp, sum[h - hbegin][1]);
          for (int w = 24; w < WOUT; ++w) {
            out[i*WOUT*WOUT + h*WOUT + w] = sum_temp[w - 24];
          }
        }

        // (3, 0) block
        hbegin = (WOUT + 1)/4*3; hend = WOUT;

        j = A->rowptr[i];
        c = _mm256_set1_ps(values[j]);
        off = A->colidx[j];

        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_mul_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD)));
          sum[h - hbegin][1] = _mm256_mul_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 8));
        }

        for (j = jbegin; j < jend; ++j) {
          c = _mm256_set1_ps(values[j]);
          off = A->colidx[j];

          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin][0] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
            sum[h - hbegin][1] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
          }
        }

        for (int h = hbegin; h < hend; ++h) {
          _mm256_store_ps(out + (i*WOUT + h)*WOUT, sum[h - hbegin][0]);
          _mm256_store_ps(out + (i*WOUT + h)*WOUT + 8, sum[h - hbegin][1]);
        }

        // (3, 1) block
        j = A->rowptr[i];
        c = _mm256_set1_ps(values[j]);
        off = A->colidx[j];

        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_mul_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 16));
          sum[h - hbegin][1] = _mm256_mul_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 24));
        }

        for (j = jbegin; j < jend; ++j) {
          c = _mm256_set1_ps(values[j]);
          off = A->colidx[j];

          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin][0] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 16), sum[h - hbegin][0]);
            sum[h - hbegin][1] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in + off + h*(WIDTH + PAD) + 24), sum[h - hbegin][1]);
          }
        }

        for (int h = hbegin; h < hend; ++h) {
          _mm256_store_ps(out + (i*WOUT + h)*WOUT + 16, sum[h - hbegin][0]);
          _mm256_store_ps(sum_temp, sum[h - hbegin][1]);
          for (int w = 24; w < WOUT; ++w) {
            out[i*WOUT*WOUT + h*WOUT + w] = sum_temp[w - 24];
          }
        }
      }
      return;
    }
    else if (width == 227 && pad == 0 && stride == 4 && k == 11) {
      int wOut = (width + 2*pad - k)/stride + 1;

      float sum[wOut];

      for (int h = 0; h < wOut; ++h) {
        //for (int w = 0; w < wOut; ++w) {
          float *in_temp = in + stride*(h*(width + pad));

          for (int i = begin; i < end; ++i) {
            for (int w = 0; w < wOut; ++w) {
              sum[w] = 0;
            }

            for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; ++j) {
              float c = values[j];
              int off = A->colidx[j];
              for (int w = 0; w < wOut; ++w) {
                sum[w] += c*in_temp[off + stride*w];
              }
            }

            for (int w = 0; w < wOut; ++w) {
              out[(i*wOut + h)*wOut + w] = sum[w];
            }
          }
        //}
      }
      return;
    }

    int wOut = (width + 2*pad - k)/stride + 1;

    if (1 == stride) {
#if 0
      float sum[wOut];

      for (int h = 0; h < wOut; ++h) {
        float *in_temp = in + h*(width + pad);

        for (int i = begin; i < end; ++i) {
          for (int w = 0; w < wOut; ++w) {
            sum[w] = 0;
          }

          for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; ++j) {
            float c = values[j];
            int off = A->colidx[j];

            for (int w = 0; w < wOut; ++w) {
              sum[w] += c*in_temp[off + w];
            }
          }

          for (int w = 0; w < wOut; ++w) {
            out[(i*wOut + h)*wOut + w] = sum[w];
          }
        }
      }
#else
      float sum[wOut*wOut];

      for (int i = begin; i < end; ++i) {
        for (int w = 0; w < wOut*wOut; ++w) {
          sum[w] = 0;
        }

        for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; ++j) {
          float c = values[j];
          int off = A->colidx[j];

          for (int h = 0; h < wOut; ++h) {
            for (int w = 0; w < wOut; ++w) {
              sum[h*wOut + w] += c*in[off + h*(width + pad) + w];
            }
          }
        }

        for (int w = 0; w < wOut*wOut; ++w) {
          out[i*wOut*wOut + w] = sum[w];
        }
      }
#endif
    }
    else {
      for (int h = 0; h < wOut; ++h) {
        for (int w = 0; w < wOut; ++w) {
          float *in_temp = in + stride*(h*(width + pad) + w);

          for (int i = begin; i < end; ++i) {
            float sum = 0;

            for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; ++j) {
              float c = values[j];
              sum += c*in_temp[A->colidx[j]];
            }

            out[(i*wOut + h)*wOut + w] = sum;
          }
        }
      }
    } // stride != 1
  }


  CSR *A;
  float *values;
  int nNonZeroKernels;

  int k; // kernel size is k*k
  int width, pad;
  int *colPerm, *colInversePerm;
};

int main(int argc, char *argv[])
{
  if (argc < 6) {
    fprintf(stderr, "Usage: %s matrix_in_matrix_market_format N(# of cols in feature matrix) (# of input channels) (image size) (stride) (pad)\n", argv[0]);
    return -1;
  }
  int N = atoi(argv[2]);
  int nic = atoi(argv[3]);
  int w = atoi(argv[4]);
  int stride = atoi(argv[5]);
  int pad = atoi(argv[6]);

  synk::Barrier::initializeInstance(omp_get_max_threads(), 1);

  // allocate a large buffer to flush out cache
  bufToFlushLlc = (double *)_mm_malloc(LLC_CAPACITY, 64);

  // Read A
  CSR *A = new CSR(argv[1]);

  // A has DP values, create a SP buffer and copy to it
  float *A_values;
  posix_memalign((void **)&A_values, 4096, sizeof(float)*A->getNnz());
  for (int i = 0; i < A->getNnz(); ++i) {
    A_values[i] = A->values[i];
  }

  // Collect all-zero columns
  CSR *AT = A->transpose();
  vector<int> nonZeroColumns; // compress -> orig col index
  vector<int> compressPerm(A->n); // orig -> compress col index
  for (int j = 0; j < A->n; ++j) {
    if (AT->rowptr[j + 1] == AT->rowptr[j]) {
      compressPerm[j] = -1;
    }
    else {
      compressPerm[j] = nonZeroColumns.size();
      nonZeroColumns.push_back(j);
    }
  }
  int nNonZeroCols = nonZeroColumns.size();

  vector<int> nonZeroRows; // compress -> orig row index
  vector<int> compressPermRow(A->m); // orig -> compress row index
  for (int i = 0; i < A->m; ++i) {
    if (A->rowptr[i + 1] == A->rowptr[i]) {
      compressPermRow[i] = -1;
    }
    else {
      compressPermRow[i] = nonZeroRows.size();
      nonZeroRows.push_back(i);
    }
  }
  int nNonZeroRows = nonZeroRows.size();
  
  KernelTensor *ATensor = new KernelTensor(A, nic, w, pad);

  printf("%s: %dx%d %d nnz (%g nnz-sparsity %g col-sparsity %g row-sparsity %g kernel-sparsity)\n", argv[1], A->m, A->n, A->getNnz(), (double)A->getNnz()/(A->m*A->n), (double)nNonZeroCols/A->n, (double)nNonZeroRows/A->m, (double)ATensor->nNonZeroKernels/(A->m*nic));
  /*for (int i = 0; i < A->m; ++i) {
    for (int j = 0; j < nic; ++j) {
      printf("%g ", (double)nnz_per_channel_pair[i][j]/kernelSize);
    }
    printf("\n");
  }*/

  // Create dense version of A
  float *A_dense;
  posix_memalign((void **)&A_dense, 4096, sizeof(float)*A->m*A->n);
  memset(A_dense, 0, sizeof(float)*A->m*A->n);
  for (int i = 0; i < A->m; ++i) {
    for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; ++j) {
      A_dense[i*A->n + A->colidx[j]] = A_values[j];
    }
  }

  // Create compressed dense version of A
  float *A_compressed;
  posix_memalign((void **)&A_compressed, 4096, sizeof(float)*A->m*nNonZeroCols);
  memset(A_compressed, 0, sizeof(float)*A->m*nNonZeroCols);
  for (int i = 0; i < A->m; ++i) {
    int r = compressPermRow[i];
    for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; ++j) {
      int c = compressPerm[A->colidx[j]];
      if (c != -1) {
        A_compressed[r*nNonZeroCols + c] = A_values[j];
      }
    }
  }
  for (int i = 0; i < A->m; ++i) {
    for (int j = 0; j < nNonZeroCols; ++j) {
      assert(A_dense[nonZeroRows[j]*A->n + nonZeroColumns[j]] == A_compressed[i*nNonZeroCols + j]);
    }
  }

  double flop = 2*A->getNnz()*N;
  double denseFlop = 2*A->m*A->n*N;
  double byte = (sizeof(float) + sizeof(int))*A->getNnz() + sizeof(int)*A->m + sizeof(float)*(nNonZeroCols + A->m)*N;

  const int NBATCH = 50;
  const int REPEAT = 16;

#ifdef NDEBUG
  double tol = 1e-1;
#else
  double tol = 1e-1; // when compiled with -O0 option, FMA is not used so less accurate
#endif
  double denseTol = 1e-1;

  // Initialize B and C
  float *B[NBATCH], *C[NBATCH], *C_ref[NBATCH];
  float *B_im[NBATCH];

  srand(0); // determinimistic randomization
  double im2col_time = 0;
  for (int b = 0; b < NBATCH; ++b) {
    posix_memalign((void **)&B[b], 4096, sizeof(float)*A->n*N);
    posix_memalign((void **)&C[b], 4096, sizeof(float)*A->m*N);
    posix_memalign((void **)&C_ref[b], 4096, sizeof(float)*A->m*N);

    posix_memalign((void **)&B_im[b], 4096, sizeof(float)*nic*(w + 2*pad)*(w + 2*pad));
    memset(B_im[b], 0, sizeof(float)*nic*(w + 2*pad)*(w + 2*pad));

    for (int i = 0; i < nic; ++i) {
      for (int j = 0; j < w; ++j) {
        for (int k = 0; k < w; ++k) {
          B_im[b][(i*(w + pad) + j + pad)*(w + pad) + k + pad] = (i + j + k + b)%17;
        }
      }
    }

    for (int i = 0; i < REPEAT; ++i) {
      flushLlc();

      im2col_time -= omp_get_wtime();
      im2col_cpu(B_im[b], nic, w, w, ATensor->k, ATensor->k, pad, pad, stride, stride, B[b], NULL, NULL);
      im2col_time += omp_get_wtime();
    }
  }
  im2col_time /= REPEAT*NBATCH;
  //printf("im2col_cpu takes %g\n", im2col_time);

  float *B_concatenated, *C_concatenated, *C_concatenated_ref;
  int N_concatenated = N*NBATCH;
  posix_memalign((void **)&B_concatenated, 4096, sizeof(float)*A->n*N*NBATCH);
  posix_memalign((void **)&C_concatenated, 4096, sizeof(float)*A->m*N*NBATCH);
  posix_memalign((void **)&C_concatenated_ref, 4096, sizeof(float)*A->m*N*NBATCH);

  for (int i = 0; i < A->n; ++i) {
    for (int b = 0; b < NBATCH; ++b) {
      for (int j = 0; j < N; ++j) {
        B_concatenated[(i*NBATCH + b)*N + j] = B[b][i*N + j];
      }
    }
  }

  const char *matdescra = "GXXCX";
  const char transa = 'N';
  float alpha = 1, beta = 0;

  double times[REPEAT*NBATCH];

  // 1. Test MKL CSRMM
  {
    for (int iter = 0; iter < REPEAT; ++iter) {
      flushLlc();

      for (int b = 0; b < NBATCH; ++b) {
        double t = omp_get_wtime();

        mkl_scsrmm(
          &transa, &A->m, &N, &A->n,
          &alpha, matdescra,
          A_values, A->colidx, A->rowptr, A->rowptr + 1,
          B[b], &N,
          &beta, C[b], &N);

        times[iter*NBATCH + b] = omp_get_wtime() - t;

        if (iter == REPEAT - 1 && b == NBATCH - 1) {
          printf("MKL_CSRMM: ");
          printEfficiency(times, REPEAT*NBATCH, flop, denseFlop, byte, im2col_time);

          // Copy reference output
          for (int b = 0; b < NBATCH; ++b) {
            for (int j = 0; j < A->m*N; ++j) {
              C_ref[b][j] = C[b][j];
            }
          }
        }
      }
    }

#if 0
    for (int iter = 0; iter < REPEAT; ++iter) {
      flushLlc();

      double t = omp_get_wtime();

      mkl_scsrmm(
        &transa, &A->m, &N_concatenated, &A->n,
        &alpha, matdescra,
        A_values, A->colidx, A->rowptr, A->rowptr + 1,
        B_concatenated, &N_concatenated,
        &beta, C_concatenated, &N_concatenated);

      times[iter] = (omp_get_wtime() - t)/NBATCH;

      if (iter == REPEAT - 1) {
        printf("MKL_CSRMM_concatenated: ");
        printEfficiency(times, REPEAT, flop, denseFlop, byte, im2col_time);

        // Copy reference output
        for (int i = 0; i < A->m*N_concatenated; ++i) {
          C_concatenated_ref[i] = C_concatenated_ref[i];
        }
      }
    }
#endif
  }

#if 0
#pragma omp parallel
  {
    int tid = omp_get_thread_num();

    for (int iter = 0; iter < REPEAT; ++iter) {
      flushLlc();

      synk::Barrier::getInstance()->wait(tid);

      double t = omp_get_wtime();

#pragma omp for nowait
      for (int b = 0; b < NBATCH; ++b) {
        mkl_scsrmm(
          &transa, &A->m, &N, &A->n,
          &alpha, matdescra,
          A_values, A->colidx, A->rowptr, A->rowptr + 1,
          B[b], &N,
          &beta, C[b], &N);
      }

      synk::Barrier::getInstance()->wait(tid);

      if (0 == tid) {
        times[iter] = (omp_get_wtime() - t)/NBATCH;

        if (iter == REPEAT - 1) {
          printf("MKL_CSRMM_parbatch: ");
          printEfficiency(times, REPEAT, flop, denseFlop, byte, im2col_time);

          for (int b = 0; b < NBATCH; ++b) {
            correctnessCheck(C_ref[b], C[b], A->m*N, tol);
            memset(C[b], 0, sizeof(float)*A->m*N);
          }
        }
      }
    }
  } // omp parallel
#endif

  // 2. Test our own CSRMM
#pragma omp parallel
  {
    int tid = omp_get_thread_num();

    for (int iter = 0; iter < REPEAT; ++iter) {
      flushLlc();

      for (int b = 0; b < NBATCH; ++b) {

        synk::Barrier::getInstance()->wait(tid);

        double t = omp_get_wtime();

        my_scsrmm(A->m, N, A->n,
          A_values, A->colidx, A->rowptr,
          B[b], N,
          C[b], N);

        synk::Barrier::getInstance()->wait(tid);

        if (0 == tid) {
          times[iter*NBATCH + b] = omp_get_wtime() - t;

          if (iter == REPEAT - 1 && b == NBATCH - 1) {
            printf("myCSRMM: ");
            printEfficiency(times, REPEAT*NBATCH, flop, denseFlop, byte, im2col_time);

            for (int b = 0; b < NBATCH; ++b) {
              correctnessCheck(C_ref[b], C[b], A->m*N, tol);
              memset(C[b], 0, sizeof(float)*A->m*N);
            }
          }
        }
      }
    }

    for (int iter = 0; iter < REPEAT; ++iter) {
      flushLlc();

      synk::Barrier::getInstance()->wait(tid);

      double t = omp_get_wtime();

      my_scsrmm(
        A->m, N_concatenated, A->n,
        A_values, A->colidx, A->rowptr,
        B_concatenated, N_concatenated,
        C_concatenated_ref, N_concatenated);

      synk::Barrier::getInstance()->wait(tid);

      if (0 == tid) {
        times[iter] = (omp_get_wtime() - t)/NBATCH;

        if (iter == REPEAT - 1) {
          printf("myCSRMM_concatenated: ");
          printEfficiency(times, REPEAT, flop, denseFlop, byte, im2col_time);

          correctnessCheck(C_concatenated_ref, C_concatenated, A->m*N*NBATCH, tol);
          memset(C_concatenated, 0, sizeof(float)*A->m*N*NBATCH);
        }
      }
    }

    for (int iter = 0; iter < REPEAT; ++iter) {
      flushLlc();

      synk::Barrier::getInstance()->wait(tid);

      double t = omp_get_wtime();

#pragma omp for nowait
      for (int b = 0; b < NBATCH; ++b) {
        my_scsrmm(A->m, N, A->n,
          A_values, A->colidx, A->rowptr,
          B[b], N,
          C[b], N,
          true);
      }

      synk::Barrier::getInstance()->wait(tid);

      if (0 == tid) {
        times[iter] = (omp_get_wtime() - t)/NBATCH;

        if (iter == REPEAT - 1) {
          printf("myCSRMM_parbatch: ");
          printEfficiency(times, REPEAT, flop, denseFlop, byte, im2col_time);

          for (int b = 0; b < NBATCH; ++b) {
            correctnessCheck(C_ref[b], C[b], A->m*N, tol);
            memset(C[b], 0, sizeof(float)*A->m*N);
          }
        }
      }
    }
  } // omp parallel

  // Convolution without lowering
#pragma omp parallel
  {
    int tid = omp_get_thread_num();

    for (int iter = 0; iter < REPEAT; ++iter) {
      flushLlc();

      for (int b = 0; b < NBATCH; ++b) {

        synk::Barrier::getInstance()->wait(tid);

        double t = omp_get_wtime();

        ATensor->conv(B_im[b], C[b], stride);

        synk::Barrier::getInstance()->wait(tid);

        if (0 == tid) {
          times[iter*NBATCH + b] = omp_get_wtime() - t;
          if (iter == REPEAT - 1 && b == NBATCH - 1) {
            printf("conv: ");
            printEfficiency(times, REPEAT*NBATCH, flop, denseFlop, byte, 0);

            for (int b = 0; b < NBATCH; ++b) {
              correctnessCheck(C_ref[b], C[b], A->m*N, tol);
              memset(C[b], 0, sizeof(float)*A->m*N);
            }
          }
        }
      }
    }

    for (int iter = 0; iter < REPEAT; ++iter) {
      flushLlc();

      synk::Barrier::getInstance()->wait(tid);

      double t = omp_get_wtime();

#pragma omp for nowait
      for (int b = 0; b < NBATCH; ++b) {
        ATensor->conv(B_im[b], C[b], stride, true);
      }

      synk::Barrier::getInstance()->wait(tid);

      if (0 == tid) {
        times[iter] = (omp_get_wtime() - t)/NBATCH;

        if (iter == REPEAT - 1) {
          printf("conv_parbatch: ");
          printEfficiency(times, REPEAT, flop, denseFlop, byte, 0);

          for (int b = 0; b < NBATCH; ++b) {
            correctnessCheck(C_ref[b], C[b], A->m*N, tol);
            memset(C[b], 0, sizeof(float)*A->m*N);
          }
        }
      }
    }
  }

  // 3-4. Test CSRMM with reordering
  {
    int *rowPerm, *rowInversePerm, *colPerm, *colInversePerm;
    posix_memalign((void **)&rowPerm, 4096, sizeof(int)*A->m);
    posix_memalign((void **)&rowInversePerm, 4096, sizeof(int)*A->m);
    posix_memalign((void **)&colPerm, 4096, sizeof(int)*A->n);
    posix_memalign((void **)&colInversePerm, 4096, sizeof(int)*A->n);
    CSR *AT = A->transpose();
    bfsBipartite(*A, *AT, rowPerm, rowInversePerm, colPerm, colInversePerm);

    FREE(A->diagptr); // A is not a matrix for linear systems, so not necessarily all diagonals are non-zeros.
    CSR *AReordered = A->permute(colPerm, rowInversePerm);
    CSR *ATReordered = AReordered->transpose();
    for (int i = 0; i < A->getNnz(); ++i) {
      A_values[i] = AReordered->values[i];
    }

    float *B_reordered[NBATCH], *C_reordered[NBATCH];

    for (int b = 0; b < NBATCH; ++b) {
      posix_memalign((void **)&B_reordered[b], 4096, sizeof(float)*A->n*N);
      posix_memalign((void **)&C_reordered[b], 4096, sizeof(float)*A->m*N);

      for (int i = 0; i < A->n; ++i) {
        for (int j = 0; j < N; ++j) {
          B_reordered[b][colPerm[i]*N + j] = B[b][i*N + j];
        }
      }
    }

    float *B_concatenated_reordered, *C_concatenated_reordered;
    posix_memalign((void **)&B_concatenated_reordered, 4096, sizeof(float)*A->n*N*NBATCH);
    posix_memalign((void **)&C_concatenated_reordered, 4096, sizeof(float)*A->m*N*NBATCH);

    for (int i = 0; i < A->n; ++i) {
      for (int j = 0; j < N*NBATCH; ++j) {
        B_concatenated_reordered[colPerm[i]*N*NBATCH + j] = B_concatenated[i*N*NBATCH + j];
      }
    }

    printf("BW is reduced by BFS reordering: %d -> %d\n", A->getBandwidth(), AReordered->getBandwidth());
    printf("Average width is reduced by BFS reordering: (%g, %g) -> (%g, %g)\n", A->getAverageWidth(), AT->getAverageWidth(), AReordered->getBandwidth(), ATReordered->getAverageWidth());

#if 0
    // 3. Test MKL CSRMM with reordering
    for (int iter = 0; iter < REPEAT; ++iter) {
      flushLlc();

      for (int b = 0; b < NBATCH; ++b) {
        double t = omp_get_wtime();

        mkl_scsrmm(
          &transa, &A->m, &N, &A->n,
          &alpha, matdescra,
          A_values, AReordered->colidx, AReordered->rowptr, AReordered->rowptr + 1,
          B_reordered[b], &N,
          &beta, C_reordered[b], &N);

        times[iter*NBATCH + b] = omp_get_wtime() - t;

        if (iter == REPEAT - 1 && b == NBATCH - 1) {
          printf("MKL_CSRMM_reordered: ");
          printEfficiency(times, REPEAT*NBATCH, flop, denseFlop, byte, im2col_time);

          for (int b = 0; b < NBATCH; ++b) {
            for (int i = 0; i < A->m; ++i) {
              for (int j = 0; j < N; ++j) {
                C[b][rowInversePerm[i]*N + j] = C_reordered[b][i*N + j];
              }
            }
            correctnessCheck(C_ref[b], C[b], AReordered->m*N, tol);
            memset(C[b], 0, sizeof(float)*AReordered->m*N);
          }
        }
      }
    }

    for (int iter = 0; iter < REPEAT; ++iter) {
      flushLlc();

      double t = omp_get_wtime();

      mkl_scsrmm(
        &transa, &A->m, &N_concatenated, &A->n,
        &alpha, matdescra,
        A_values, AReordered->colidx, AReordered->rowptr, AReordered->rowptr + 1,
        B_concatenated_reordered, &N_concatenated,
        &beta, C_concatenated_reordered, &N_concatenated);

      times[iter] = (omp_get_wtime() - t)/NBATCH;

      if (iter == REPEAT - 1) {
        printf("MKL_CSRMM_reordered_concatenated: ");
        printEfficiency(times, REPEAT, flop, denseFlop, byte, im2col_time);

        for (int i = 0; i < A->m; ++i) {
          for (int j = 0; j < N*NBATCH; ++j) {
            C_concatenated[rowInversePerm[i]*N*NBATCH + j] = C_concatenated_reordered[i*N*NBATCH + j];
          }
        }
        correctnessCheck(C_concatenated_ref, C_concatenated, AReordered->m*N*NBATCH, tol);
        memset(C_concatenated, 0, sizeof(float)*AReordered->m*N*NBATCH);
      }
    }

#pragma omp parallel
    {
      int tid = omp_get_thread_num();

      for (int iter = 0; iter < REPEAT; ++iter) {
        flushLlc();

        synk::Barrier::getInstance()->wait(tid);

        double t = omp_get_wtime();

#pragma omp for nowait
        for (int b = 0; b < NBATCH; ++b) {
          mkl_scsrmm(
            &transa, &A->m, &N, &A->n,
            &alpha, matdescra,
            A_values, AReordered->colidx, AReordered->rowptr, AReordered->rowptr + 1,
            B_reordered[b], &N,
            &beta, C_reordered[b], &N);
        }

        synk::Barrier::getInstance()->wait(tid);

        if (0 == tid) {
          times[iter] = (omp_get_wtime() - t)/NBATCH;

          if (iter == REPEAT - 1) {
            printf("MKL_CSRMM_reordered_parbatch: ");
            printEfficiency(times, REPEAT, flop, denseFlop, byte, im2col_time);
            for (int b = 0; b < NBATCH; ++b) {
              for (int i = 0; i < A->m; ++i) {
                for (int j = 0; j < N; ++j) {
                  C[b][rowInversePerm[i]*N + j] = C_reordered[b][i*N + j];
                }
              }
              correctnessCheck(C_ref[b], C[b], AReordered->m*N, tol);
              memset(C[b], 0, sizeof(float)*AReordered->m*N);
            }
          }
        }
      }
    } // omp parallel
#endif

    // 4. Test our own CSRMM with reordering
#pragma omp parallel
    {
      int tid = omp_get_thread_num();

      for (int iter = 0; iter < REPEAT; ++iter) {
        flushLlc();

        for (int b = 0; b < NBATCH; ++b) {
          synk::Barrier::getInstance()->wait(tid);

          double t = omp_get_wtime();

          my_scsrmm(AReordered->m, N, AReordered->n,
            A_values, AReordered->colidx, AReordered->rowptr,
            B_reordered[b], N,
            C_reordered[b], N);

          synk::Barrier::getInstance()->wait(tid);

          if (0 == tid) {
            times[iter*NBATCH + b] = omp_get_wtime() - t;

            if (iter == REPEAT - 1 && b == NBATCH - 1) {
              printf("myCSRMM_reordered: ");
              printEfficiency(times, REPEAT*NBATCH, flop, denseFlop, byte, im2col_time);

              for (int b = 0; b < NBATCH; ++b) {
                for (int i = 0; i < A->m; ++i) {
                  for (int j = 0; j < N; ++j) {
                    C[b][rowInversePerm[i]*N + j] = C_reordered[b][i*N + j];
                  }
                }
                correctnessCheck(C_ref[b], C[b], AReordered->m*N, tol);
                memset(C[b], 0, sizeof(float)*AReordered->m*N);
              }
            }
          }
        }
      }

      for (int iter = 0; iter < REPEAT; ++iter) {
        flushLlc();

        synk::Barrier::getInstance()->wait(tid);

        double t = omp_get_wtime();

        my_scsrmm(
          AReordered->m, N*NBATCH, AReordered->n,
          A_values, AReordered->colidx, AReordered->rowptr,
          B_concatenated_reordered, N*NBATCH,
          C_concatenated_reordered, N*NBATCH);

        synk::Barrier::getInstance()->wait(tid);

        if (0 == tid) {
          times[iter] = (omp_get_wtime() - t)/NBATCH;

          if (iter == REPEAT - 1) {
            printf("myCSRMM_reordered_concatenated: ");
            printEfficiency(times, REPEAT, flop, denseFlop, byte, im2col_time);

            for (int i = 0; i < A->m; ++i) {
              for (int j = 0; j < N*NBATCH; ++j) {
                C_concatenated[rowInversePerm[i]*N*NBATCH + j] = C_concatenated_reordered[i*N*NBATCH + j];
              }
            }

            correctnessCheck(C_concatenated_ref, C_concatenated, AReordered->m*N*NBATCH, tol);
            memset(C_concatenated, 0, sizeof(float)*AReordered->m*N*NBATCH);
          }
        }
      }

      for (int iter = 0; iter < REPEAT; ++iter) {
        flushLlc();

        synk::Barrier::getInstance()->wait(tid);

        double t = omp_get_wtime();

#pragma omp for nowait
        for (int b = 0; b < NBATCH; ++b) {
          my_scsrmm(
            AReordered->m, N, AReordered->n,
            A_values, AReordered->colidx, AReordered->rowptr,
            B_reordered[b], N,
            C_reordered[b], N,
            true);
        }

        synk::Barrier::getInstance()->wait(tid);

        if (0 == tid) {
          times[iter] = (omp_get_wtime() - t)/NBATCH;

          if (iter == REPEAT - 1) {
            printf("myCSRMM_reordered_parbatch: ");
            printEfficiency(times, REPEAT, flop, denseFlop, byte, im2col_time);

            for (int b = 0; b < NBATCH; ++b) {
              for (int i = 0; i < A->m; ++i) {
                for (int j = 0; j < N; ++j) {
                  C[b][rowInversePerm[i]*N + j] = C_reordered[b][i*N + j];
                }
              }
              correctnessCheck(C_ref[b], C[b], AReordered->m*N, tol);
              memset(C[b], 0, sizeof(float)*AReordered->m*N);
            }
          }
        }
      }
    } // omp parallel

    delete AT;
    delete AReordered;
    delete ATReordered;
  }

  // 5. Test dense SGEMM
  {
    for (int iter = 0; iter < REPEAT; ++iter) {
      flushLlc();

      for (int b = 0; b < NBATCH; ++b) {

        double t = omp_get_wtime();

        cblas_sgemm(
          CblasRowMajor, CblasNoTrans, CblasNoTrans, 
          A->m, N, A->n,
          alpha, A_dense, A->n,
          B[b], N,
          beta, C[b], N);

        times[iter*NBATCH + b] = omp_get_wtime() - t;

        if (iter == REPEAT - 1 && b == NBATCH - 1) {
          printf("dense: ");
          printEfficiency(times, REPEAT*NBATCH, flop, denseFlop, byte, im2col_time);

          for (int b = 0; b < NBATCH; ++b) {
            correctnessCheck(C_ref[b], C[b], A->m*N, denseTol);
            memset(C[b], 0, sizeof(float)*A->m*N);
          }
        }
      }
    }

    for (int iter = 0; iter < REPEAT; ++iter) {
      flushLlc();

      double t = omp_get_wtime();

      cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        A->m, N_concatenated, A->n,
        alpha, A_dense, A->n,
        B_concatenated, N_concatenated,
        beta, C_concatenated, N_concatenated);

      times[iter] = (omp_get_wtime() - t)/NBATCH;

      if (iter == REPEAT - 1) {
        printf("dense_concatenated: ");
        printEfficiency(times, REPEAT, flop, denseFlop, byte, im2col_time);

        correctnessCheck(C_concatenated_ref, C_concatenated, A->m*N*NBATCH, denseTol);
        memset(C_concatenated, 0, sizeof(float)*A->m*N*NBATCH);
      }
    }
  }

#pragma omp parallel
  {
    int tid = omp_get_thread_num();

    for (int iter = 0; iter < REPEAT; ++iter) {
      flushLlc();

      synk::Barrier::getInstance()->wait(tid);

      double t = omp_get_wtime();

#pragma omp for nowait
      for (int b = 0; b < NBATCH; ++b) {
        cblas_sgemm(
          CblasRowMajor, CblasNoTrans, CblasNoTrans, 
          A->m, N, A->n,
          alpha, A_dense, A->n,
          B[b], N,
          beta, C[b], N);
      }

      synk::Barrier::getInstance()->wait(tid);

      if (0 == tid) {
        times[iter] = (omp_get_wtime() - t)/NBATCH;

        if (iter == REPEAT - 1) {
          printf("dense_parbatch: ");
          printEfficiency(times, REPEAT, flop, denseFlop, byte, im2col_time);

          for (int b = 0; b < NBATCH; ++b) {
            correctnessCheck(C_ref[b], C[b], A->m*N, denseTol);
            memset(C[b], 0, sizeof(float)*A->m*N);
          }
        }
      }
    }
  } // omp parallel

  // 6. Test compressed dense SGEMM
  float *B_compressed[NBATCH];
  for (int b = 0; b < NBATCH; ++b) {
    posix_memalign((void **)&B_compressed[b], 4096, sizeof(float)*nNonZeroCols*N);

    for (int i = 0; i < nNonZeroCols; ++i) {
      int r = nonZeroColumns[i];
      for (int j = 0; j < N; ++j) {
        B_compressed[b][i*N + j] = B[b][r*N + j];
      }
    }
  }

  float *B_concatenated_compressed;
  posix_memalign((void **)&B_concatenated_compressed, 4096, sizeof(float)*nNonZeroCols*N*NBATCH);
  for (int i = 0; i < nNonZeroCols; ++i) {
    int r = nonZeroColumns[i];
    for (int j = 0; j < N*NBATCH; ++j) {
      B_concatenated_compressed[i*N*NBATCH + j] = B_concatenated[r*N*NBATCH + j];
    }
  }

  float *C_compressed[NBATCH];
  for (int b = 0; b < NBATCH; ++b) {
    posix_memalign((void **)&C_compressed[b], 4096, sizeof(float)*nNonZeroRows*N);
  }
  float *C_concatenated_compressed;
  posix_memalign((void **)&C_concatenated_compressed, 4096, sizeof(float)*nNonZeroRows*N*NBATCH);

  {
    for (int iter = 0; iter < REPEAT; ++iter) {
      flushLlc();

      for (int b = 0; b < NBATCH; ++b) {
        double t = omp_get_wtime();

        cblas_sgemm(
          CblasRowMajor, CblasNoTrans, CblasNoTrans, 
          nNonZeroRows, N, nNonZeroCols,
          alpha, A_compressed, nNonZeroCols,
          B_compressed[b], N,
          beta, C_compressed[b], N);

        times[iter*NBATCH + b] = omp_get_wtime() - t;

        if (iter == REPEAT - 1 && b == NBATCH - 1) {
          printf("compressed: ");
          printEfficiency(times, REPEAT*NBATCH, flop, denseFlop, byte, im2col_time);

          for (int b = 0; b < NBATCH; ++b) {
            for (int i = 0; i < nNonZeroRows; ++i) {
              memcpy(C[b] + nonZeroRows[i]*N, C_compressed[b] + i*N, sizeof(float)*N);
            }
            correctnessCheck(C_ref[b], C[b], A->m*N, denseTol);
            memset(C[b], 0, sizeof(float)*A->m*N);
          }
        }
      }
    }

    for (int iter = 0; iter < REPEAT; ++iter) {
      flushLlc();

      double t = omp_get_wtime();

      cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        nNonZeroRows, N_concatenated, nNonZeroCols,
        alpha, A_compressed, nNonZeroCols,
        B_concatenated_compressed, N_concatenated,
        beta, C_concatenated_compressed, N_concatenated);

      times[iter] = (omp_get_wtime() - t)/NBATCH;

      if (iter == REPEAT - 1) {
        printf("compressed_concatenated: ");
        printEfficiency(times, REPEAT, flop, denseFlop, byte, im2col_time);

        for (int i = 0; i < nNonZeroRows; ++i) {
          memcpy(C_concatenated + nonZeroRows[i]*N_concatenated, C_concatenated_compressed + i*N_concatenated, sizeof(float)*N_concatenated);
        }
        correctnessCheck(C_concatenated_ref, C_concatenated, A->m*N*NBATCH, denseTol);
        memset(C_concatenated, 0, sizeof(float)*A->m*N*NBATCH);
      }
    }
  }

#pragma omp parallel
  {
    int tid = omp_get_thread_num();

    for (int iter = 0; iter < REPEAT; ++iter) {
      flushLlc();

      synk::Barrier::getInstance()->wait(tid);

      double t = omp_get_wtime();

#pragma omp for nowait
      for (int b = 0; b < NBATCH; ++b) {
#ifdef USE_LIBXSMM
        libxsmm_sgemm(
          NULL, NULL,
          &nNonZeroRows, &N, &nNonZeroCols, 
          &alpha, A_compressed, NULL,
          B_compressed[b], NULL,
          &beta, C_compressed[b], NULL);
#else
        cblas_sgemm(
          CblasRowMajor, CblasNoTrans, CblasNoTrans, 
          nNonZeroRows, N, nNonZeroCols,
          alpha, A_compressed, nNonZeroCols,
          B_compressed[b], N,
          beta, C_compressed[b], N);
#endif
      }

      synk::Barrier::getInstance()->wait(tid);

      if (0 == tid) {
        times[iter] = (omp_get_wtime() - t)/NBATCH;

        if (iter == REPEAT - 1) {
          printf("compressed_parbatch: ");
          printEfficiency(times, REPEAT, flop, denseFlop, byte, im2col_time);

          for (int b = 0; b < NBATCH; ++b) {
            for (int i = 0; i < nNonZeroRows; ++i) {
              memcpy(C[b] + nonZeroRows[i]*N, C_compressed[b] + i*N, sizeof(float)*N);
            }
            correctnessCheck(C_ref[b], C[b], A->m*N, denseTol);
            memset(C[b], 0, sizeof(float)*A->m*N);
          }
        }
      }
    }
  } // omp parallel

  for (int b = 0; b < NBATCH; ++b) {
    free(B_compressed[b]);
  }

  delete A;
  delete AT;

  free(A_values);
  free(A_dense);
  free(A_compressed);

  for (int b = 0; b < NBATCH; ++b) {
    free(B[b]);
    free(C[b]);
    free(C_ref[b]);
  }

  return 0;
}
