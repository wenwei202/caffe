/*
 * spgemm.hpp
 *
 *  Created on: May 13, 2016
 *      Author: jpark103
 */

#ifndef CAFFE_UTIL_SPGEMM_HPP_
#define CAFFE_UTIL_SPGEMM_HPP_

#include <map>
#include <string>
#include <omp.h>
#include <immintrin.h>

struct CSR
{
  float *values;
  int *colidx;
  int *rowptr;
  int m, n;
};

extern std::map<std::string, CSR> layer2weight;
extern std::map<std::string, float *> layer2bottom;
extern std::map<std::string, float *> layer2bias;

static int spgemm_flops(
    const float *A_data, const int *A_j, const int *A_i,
    const float *B_data, const int *B_j, const int *B_i,
    int m)
{
  int flops = 0;
#pragma omp parallel for reduction(+:flops)
  for (int i = 0; i < m; ++i) {
    for (int j = A_i[i]; j < A_i[i + 1]; ++j) {
      int ja = A_j[j];
      flops += 2*(B_i[ja + 1] - B_i[ja]);
    }
  }
  return flops;
}

static void csrmultd(
    const float *A_data, const int *A_j, const int *A_i,
    const float *B_data, const int *B_j, const int *B_i,
    const float *bias,
    float *C,
    int m, int n)
{
#pragma omp parallel for
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      C[i*n + j] = bias[j];
    }
    for (int j = A_i[i]; j < A_i[i + 1]; ++j) {
      int ja = A_j[j];
      float a_entry = A_data[j];
      for (int k = B_i[ja]; k < B_i[ja + 1]; ++k) {
        int jb = B_j[k];
        float b_entry = B_data[k];
        C[i*n + jb] += a_entry*b_entry;
      }
    }
    for (int j = 0; j < n; ++j) {
      C[i*n + j] = std::max<float>(0, C[i*n + j]);
    }
  }
}

static int csrmultd_fused_flops(
    const float *A,
    const float *B_data, const int *B_j, const int *B_i,
    const float *C_data, const int *C_j, const int *C_i,
    const float *D_data, const int *D_j, const int *D_i,
    const float *B_bias, const float *C_bias, const float *D_bias,
    float *E,
    int A_num_rows,
    int A_num_cols, int B_num_cols, int C_num_cols, int D_num_cols,
    float *B_temp_global, float *C_temp_global)
{
  int flops = 0;

#pragma omp parallel reduction(+:flops)
  {
    int tid = omp_get_thread_num();
    float *B_temp = B_temp_global + tid*B_num_cols;
    float *C_temp = C_temp_global + tid*C_num_cols;

#pragma omp for
    for (int i = 0; i < A_num_rows; ++i) {
      for (int j = 0; j < B_num_cols; ++j) {
        B_temp[j] = B_bias[j];
      }
      for (int j = 0; j < A_num_cols; ++j) {
        float a_entry = A[i*A_num_cols + j];
        if (a_entry == 0) continue;
        for (int k = B_i[j]; k < B_i[j + 1]; ++k) {
          B_temp[B_j[k]] += a_entry*B_data[k];
          flops += 2;
        }
      }

      for (int j = 0; j < C_num_cols; ++j) {
        C_temp[j] = C_bias[j];
      }
      for (int j = 0; j < B_num_cols; ++j) {
        float b_entry = B_temp[j];
        if (b_entry <= 0) continue;
        for (int k = C_i[j]; k < C_i[j + 1]; ++k) {
          C_temp[C_j[k]] += b_entry*C_data[k];
          flops += 2;
        }
      }

      for (int j = 0; j < D_num_cols; ++j) {
        E[i*D_num_cols + j] = D_bias[j];
      }
      for (int j = 0; j < C_num_cols; ++j) {
        float c_entry = C_temp[j];
        if (c_entry <= 0) continue;
        for (int k = D_i[j]; k < D_i[j + 1]; ++k) {
          E[i*D_num_cols + D_j[k]] += c_entry*D_data[k];
          flops += 2;
        }
      }
    }
  }

  return flops;
}

/* E = A*B*C*D */
static void csrmultd_fused(
    const float *A,
    const float *B_data, const int *B_j, const int *B_i,
    const float *C_data, const int *C_j, const int *C_i,
    const float *D_data, const int *D_j, const int *D_i,
    const float *B_bias, const float *C_bias, const float *D_bias,
    float *E,
    int A_num_rows,
    int A_num_cols, int B_num_cols, int C_num_cols, int D_num_cols,
    float *B_temp_global, float *C_temp_global)
{
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    float *B_temp = B_temp_global + tid*B_num_cols;
    float *C_temp = C_temp_global + tid*C_num_cols;

//    for (int j = 0; j < B_num_cols; ++j) {
//      B_temp[j] = B_bias[j];
//    }
//    for (int j = 0; j < C_num_cols; ++j) {
//      C_temp[j] = C_bias[j];
//    }

#pragma omp for
    for (int i = 0; i < A_num_rows; ++i) {
      for (int j = 0; j < B_num_cols; ++j) {
        B_temp[j] = B_bias[j];
      }
      for (int j = 0; j < A_num_cols; ++j) {
        float a_entry = A[i*A_num_cols + j];
        if (a_entry == 0) continue;
        for (int k = B_i[j]; k < B_i[j + 1]; ++k) {
          B_temp[B_j[k]] += a_entry*B_data[k];
        }
      }

      for (int j = 0; j < C_num_cols; ++j) {
        C_temp[j] = C_bias[j];
      }
      for (int j = 0; j < B_num_cols; ++j) {
        float b_entry = B_temp[j];
//        B_temp[j] = B_bias[j];
        if (b_entry <= 0) continue;
        for (int k = C_i[j]; k < C_i[j + 1]; ++k) {
          C_temp[C_j[k]] += b_entry*C_data[k];
        }
      }

      for (int j = 0; j < D_num_cols; ++j) {
        E[i*D_num_cols + j] = D_bias[j];
      }
      for (int j = 0; j < C_num_cols; ++j) {
        float c_entry = C_temp[j];
//        C_temp[j] = C_bias[j];
        if (c_entry <= 0) continue;
        for (int k = D_i[j]; k < D_i[j + 1]; ++k) {
          E[i*D_num_cols + D_j[k]] += c_entry*D_data[k];
        }
      }
    }
  }
}

static void spgemm(
    const float *A_data, const int *A_j, const int *A_i,
    const float *B_data, const int *B_j, const int *B_i,
    const float *bias,
    float *C_data, int *C_j, int *C_i, int *cnnz,
    int m, int n, float *x)
{
  for (int j = 0; j < n; ++j) {
    x[j] = bias[j];
  }

  int nnz = 0;
  C_i[0] = 0;
  for (int i = 0; i < m; ++i) {
    for (int j = A_i[i]; j < A_i[i + 1]; ++j) {
      int ja = A_j[j];
      float a_entry = A_data[j];
      for (int k = B_i[ja]; k < B_i[ja + 1]; ++k) {
        int jb = B_j[k];
        float b_entry = B_data[k];
        x[jb] += a_entry*b_entry;
      }
    }

    for (int j = 0; j < n; ++j) {
      if (x[j] > 0) {
        C_j[nnz] = j;
        C_data[nnz] = x[j];
        ++nnz;
      }
      x[j] = bias[j];
    }
    C_i[i + 1] = nnz;
  }

  *cnnz = nnz;
}

static void csrmultd_csc(
    const float *A_data, const int *A_j, const int *A_i,
    const float *B_data, const int *B_j, const int *B_i,
    const float *bias,
    float *C,
    int m, int n)
{
//  for (int i = 0; i < m; ++i) {
//    printf("%d:%d\n", i, A_i[i + 1] - A_i[i]);
//  }
#pragma omp parallel for
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      C[i*n + j] = bias[i];
    }
    for (int j = A_i[i]; j < A_i[i + 1]; ++j) {
      int ja = A_j[j];
      float a_entry = A_data[j];
      for (int k = B_i[ja]; k < B_i[ja + 1]; ++k) {
        int jb = B_j[k];
        float b_entry = B_data[k];
        C[i*n + jb] += a_entry*b_entry;
      }
    }
    for (int j = 0; j < n; ++j) {
      C[i*n + j] = std::max<float>(0, C[i*n + j]);
    }
  }
}

// C = A*B
static void __attribute__((noinline)) csrmm_fused(
    const float *A_data, const int *A_j, const int *A_i,
    const float *B,
    float *C,
    int m, int n, int k,
    const float *bias,
    int col_block_size)
{
  int ncolblocks = k/col_block_size;

#pragma omp parallel
  {
    int cb = 0;
#pragma omp for nowait
    for (int i = 0; i < m; ++i) {
#ifdef __AVX2__
      __m256 sum[8];
      for (int kk = 0; kk < n; kk += 64) { // assume n (batch size) is a multiple of 64
        for (int k = 0; k < 8; ++k) {
          sum[k] = _mm256_set1_ps(bias[i]);
        }
        for (int j = A_i[cb*m + i]; j < A_i[cb*m + i + 1]; ++j) {
          int c = A_j[j];
          __m256 v_v = _mm256_set1_ps(A_data[j]);
          for (int k = 0; k < 8; ++k) {
            sum[k] = _mm256_fmadd_ps(v_v, _mm256_load_ps(B + c + kk + k*8), sum[k]);
          }
        }
        for (int k = 0; k < 8; ++k) {
          _mm256_store_ps(C + i*n + kk + k*8, sum[k]);
        }
      }
#else
      __m128 sum[8];
      for (int kk = 0; kk < n; kk += 32) { // assume n (batch size) is a multiple of 32
        for (int k = 0; k < 8; ++k) {
          sum[k] = _mm_set1_ps(bias[i]);
        }
        for (int j = A_i[cb*m + i]; j < A_i[cb*m + i + 1]; ++j) {
          int c = A_j[j];
          __m128 v_v = _mm_set1_ps(A_data[j]);
          for (int k = 0; k < 8; ++k) {
            sum[k] = _mm_add_ps(_mm_mul_ps(v_v, _mm_load_ps(B + c + kk + k*4)), sum[k]);
          }
        }
        for (int k = 0; k < 8; ++k) {
          _mm_store_ps(C + i*n + kk + k*4, sum[k]);
        }
      }
#endif
    }

    for (cb = 1; cb < ncolblocks - 1; ++cb) {
#pragma omp for nowait
      for (int i = 0; i < m; ++i) {
#ifdef __AVX2__
        __m256 sum[8];
        for (int kk = 0; kk < n; kk += 64) {
          for (int k = 0; k < 8; ++k) {
            sum[k] = _mm256_load_ps(C + i*n + kk + k*8);
          }
          for (int j = A_i[cb*m + i]; j < A_i[cb*m + i + 1]; ++j) {
            int c = A_j[j];
            __m256 v_v = _mm256_set1_ps(A_data[j]);
            for (int k = 0; k < 8; ++k) {
              sum[k] = _mm256_fmadd_ps(v_v, _mm256_load_ps(B + c + kk + k*8), sum[k]);
            }
          }
          for (int k = 0; k < 8; ++k) {
            _mm256_store_ps(C + i*n + kk + k*8, sum[k]);
          }
        }
#else
        __m128 sum[8];
        for (int kk = 0; kk < n; kk += 32) { // assume n (batch size) is a multiple of 32
          for (int k = 0; k < 8; ++k) {
            sum[k] = _mm_load_ps(C + i*n + kk + k*4);
          }
          for (int j = A_i[cb*m + i]; j < A_i[cb*m + i + 1]; ++j) {
            int c = A_j[j];
            __m128 v_v = _mm_set1_ps(A_data[j]);
            for (int k = 0; k < 8; ++k) {
              sum[k] = _mm_add_ps(_mm_mul_ps(v_v, _mm_load_ps(B + c + kk + k*4)), sum[k]);
            }
          }
          for (int k = 0; k < 8; ++k) {
            _mm_store_ps(C + i*n + kk + k*4, sum[k]);
          }
        }
#endif
      }
    } // for each col block

#pragma omp for nowait
    for (int i = 0; i < m; ++i) {
#ifdef __AVX2__
      __m256 sum[8];
      for (int kk = 0; kk < n; kk += 64) {
        for (int k = 0; k < 8; ++k) {
          sum[k] = _mm256_load_ps(C + i*n + kk + k*8);
        }
        for (int j = A_i[cb*m + i]; j < A_i[cb*m + i + 1]; ++j) {
          int c = A_j[j];
          __m256 v_v = _mm256_set1_ps(A_data[j]);
          for (int k = 0; k < 8; ++k) {
            sum[k] = _mm256_fmadd_ps(v_v, _mm256_load_ps(B + c + kk + k*8), sum[k]);
          }
        }
        for (int k = 0; k < 8; ++k) {
          _mm256_store_ps(C + i*n + kk + k*8, _mm256_max_ps(_mm256_setzero_ps(), sum[k]));
        }
      }
#else
      __m128 sum[8];
      for (int kk = 0; kk < n; kk += 32) { // assume n (batch size) is a multiple of 32
        for (int k = 0; k < 8; ++k) {
          sum[k] = _mm_load_ps(C + i*n + kk + k*4);
        }
        for (int j = A_i[cb*m + i]; j < A_i[cb*m + i + 1]; ++j) {
          int c = A_j[j];
          __m128 v_v = _mm_set1_ps(A_data[j]);
          for (int k = 0; k < 8; ++k) {
            sum[k] = _mm_add_ps(_mm_mul_ps(v_v, _mm_load_ps(B + c + kk + k*4)), sum[k]);
          }
        }
        for (int k = 0; k < 8; ++k) {
          _mm_store_ps(C + i*n + kk + k*4, _mm_max_ps(_mm_setzero_ps(), sum[k]));
        }
      }
#endif
    }
  } // omp parallel
}

static void __attribute__((noinline)) csrmm(
    const float *A_data, const int *A_j, const int *A_i,
    const float *B,
    float *C,
    int m, int n, int k,
    const float *bias,
    int col_block_size)
{
  int ncolblocks = k/col_block_size;

#pragma omp parallel
  {
    int cb = 0;
#pragma omp for nowait
    for (int i = 0; i < m; ++i) {
#ifdef __AVX2__
      __m256 sum[8];
      for (int kk = 0; kk < n; kk += 64) { // assume n (batch size) is a multiple of 64
        for (int k = 0; k < 8; ++k) {
          sum[k] = _mm256_set1_ps(bias[i]);
        }
        for (int j = A_i[cb*m + i]; j < A_i[cb*m + i + 1]; ++j) {
          int c = A_j[j];
          __m256 v_v = _mm256_set1_ps(A_data[j]);
          for (int k = 0; k < 8; ++k) {
            sum[k] = _mm256_fmadd_ps(v_v, _mm256_load_ps(B + c + kk + k*8), sum[k]);
          }
        }
        for (int k = 0; k < 8; ++k) {
          _mm256_store_ps(C + i*n + kk + k*8, sum[k]);
        }
      }
#else
      __m128 sum[8];
      for (int kk = 0; kk < n; kk += 32) { // assume n (batch size) is a multiple of 32
        for (int k = 0; k < 8; ++k) {
          sum[k] = _mm_set1_ps(bias[i]);
        }
        for (int j = A_i[cb*m + i]; j < A_i[cb*m + i + 1]; ++j) {
          int c = A_j[j];
          __m128 v_v = _mm_set1_ps(A_data[j]);
          for (int k = 0; k < 8; ++k) {
            sum[k] = _mm_add_ps(_mm_mul_ps(v_v, _mm_load_ps(B + c + kk + k*4)), sum[k]);
          }
        }
        for (int k = 0; k < 8; ++k) {
          _mm_store_ps(C + i*n + kk + k*4, sum[k]);
        }
      }
#endif
    }

    for (cb = 1; cb < ncolblocks; ++cb) {
#pragma omp for nowait
      for (int i = 0; i < m; ++i) {
#ifdef __AVX2__
        __m256 sum[8];
        for (int kk = 0; kk < n; kk += 64) {
          for (int k = 0; k < 8; ++k) {
            sum[k] = _mm256_load_ps(C + i*n + kk + k*8);
          }
          for (int j = A_i[cb*m + i]; j < A_i[cb*m + i + 1]; ++j) {
            int c = A_j[j];
            __m256 v_v = _mm256_set1_ps(A_data[j]);
            for (int k = 0; k < 8; ++k) {
              sum[k] = _mm256_fmadd_ps(v_v, _mm256_load_ps(B + c + kk + k*8), sum[k]);
            }
          }
          for (int k = 0; k < 8; ++k) {
            _mm256_store_ps(C + i*n + kk + k*8, sum[k]);
          }
        }
#else
        __m128 sum[8];
        for (int kk = 0; kk < n; kk += 32) { // assume n (batch size) is a multiple of 32
          for (int k = 0; k < 8; ++k) {
            sum[k] = _mm_load_ps(C + i*n + kk + k*4);
          }
          for (int j = A_i[cb*m + i]; j < A_i[cb*m + i + 1]; ++j) {
            int c = A_j[j];
            __m128 v_v = _mm_set1_ps(A_data[j]);
            for (int k = 0; k < 8; ++k) {
              sum[k] = _mm_add_ps(_mm_mul_ps(v_v, _mm_load_ps(B + c + kk + k*4)), sum[k]);
            }
          }
          for (int k = 0; k < 8; ++k) {
            _mm_store_ps(C + i*n + kk + k*4, sum[k]);
          }
        }
#endif
      }
    } // for each col block
  } // omp parallel
}

static void spgemm_csc(
    const float *A_data, const int *A_j, const int *A_i,
    const float *B_data, const int *B_j, const int *B_i,
    const float *bias,
    float *C_data, int *C_j, int *C_i, int *cnnz,
    int m, int n, float *x)
{
  for (int j = 0; j < n; ++j) {
    x[j] = 0;
  }

  int nnz = 0;
  C_i[0] = 0;
  for (int i = 0; i < m; ++i) {
    for (int j = A_i[i]; j < A_i[i + 1]; ++j) {
      int ja = A_j[j];
      float a_entry = A_data[j];
      for (int k = B_i[ja]; k < B_i[ja + 1]; ++k) {
        int jb = B_j[k];
        float b_entry = B_data[k];
        x[jb] += a_entry*b_entry;
      }
    }

    for (int j = 0; j < n; ++j) {
      if (bias[i] + x[j] > 0) {
        C_j[nnz] = j;
        C_data[nnz] = bias[i] + x[j];
        ++nnz;
      }
      x[j] = 0;
    }
    C_i[i + 1] = nnz;
  }

  *cnnz = nnz;
}

#endif /* CAFFE_UTIL_SPGEMM_HPP_ */
