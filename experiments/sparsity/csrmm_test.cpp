#include <stdlib.h>
#include <string.h>

#include <vector>

#include <omp.h>
#include <mkl.h>

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
  double *times, int REPEAT, double flop, double denseFlop, double byte)
{
  sort(times, times + REPEAT);

  double t = times[REPEAT/2];

  printf(
    "%7.2f sparse_gflops %7.2f dense_gflops %7.2f sparse_gbps\n",
    flop/t/1e9, denseFlop/t/1e9, byte/t/1e9);
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
      }
    }
//#pragma vector nontemporal(C)
    for (int k = 0; k < N; ++k) {
      C[i*N + k] = sum[k];
    }
  }
#endif
}

int my_sgemm(
  int M, int N, int K,
  const float *A, const float *B, float *C)
{
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      double sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += A[i*K + k]*B[k*N + j];
      }
      C[i*N + j] = sum;
    }
  }
}

int main(int argc, char *argv[])
{
  if (argc < 2) {
    fprintf(stderr, "Usage: %s matrix_in_matrix_market_format N(# of cols in feature matrix)\n", argv[0]);
    return -1;
  }
  int N = atoi(argv[2]);

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
  printf("%s: %dx%d %d nnz (%g nnz-sparsity %g col-sparsity)\n", argv[1], A->m, A->n, A->getNnz(), (double)A->getNnz()/(A->m*A->n), (double)nNonZeroCols/A->n);

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
    for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; ++j) {
      int c = compressPerm[A->colidx[j]];
      if (c != -1) {
        A_compressed[i*nNonZeroCols + c] = A_values[j];
      }
    }
  }
  for (int i = 0; i < A->m; ++i) {
    for (int j = 0; j < nNonZeroCols; ++j) {
      assert(A_dense[i*A->n + nonZeroColumns[j]] == A_compressed[i*nNonZeroCols + j]);
    }
  }

  double flop = 2*A->getNnz()*N;
  double denseFlop = 2*A->m*A->n*N;
  double byte = (sizeof(float) + sizeof(int))*A->getNnz() + sizeof(int)*A->m + sizeof(float)*(nNonZeroCols + A->m)*N;

  const int NBATCH = 50;
  const int REPEAT = 16;

#ifdef NDEBUG
  double tol = 1e-8;
#else
  double tol = 1e-1; // when compiled with -O0 option, FMA is not used so less accurate
#endif
  double denseTol = 1e-1;

  // Initialize B and C
  float *B[NBATCH], *C[NBATCH], *C_ref[NBATCH];

  srand(0); // determinimistic randomization
  for (int b = 0; b < NBATCH; ++b) {
    posix_memalign((void **)&B[b], 4096, sizeof(float)*A->n*N);
    posix_memalign((void **)&C[b], 4096, sizeof(float)*A->m*N);
    posix_memalign((void **)&C_ref[b], 4096, sizeof(float)*A->m*N);

    for (int j = 0; j < A->n*N; ++j) {
      B[b][j] = j;//rand();
    }
  }

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
          printEfficiency(times, REPEAT*NBATCH, flop, denseFlop, byte);

          // Copy reference output
          for (int b = 0; b < NBATCH; ++b) {
            for (int j = 0; j < A->m*N; ++j) {
              C_ref[b][j] = C[b][j];
            }
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
        A_values, A->colidx, A->rowptr, A->rowptr + 1,
        B_concatenated, &N_concatenated,
        &beta, C_concatenated, &N_concatenated);

      times[iter] = omp_get_wtime() - t;

      if (iter == REPEAT - 1) {
        printf("MKL_CSRMM_concatenated: ");
        printEfficiency(times, REPEAT, flop*NBATCH, denseFlop*NBATCH, byte*NBATCH);

        // Copy reference output
        for (int i = 0; i < A->m*N_concatenated; ++i) {
          C_concatenated_ref[i] = C_concatenated_ref[i];
        }
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
        mkl_scsrmm(
          &transa, &A->m, &N, &A->n,
          &alpha, matdescra,
          A_values, A->colidx, A->rowptr, A->rowptr + 1,
          B[b], &N,
          &beta, C[b], &N);
      }

      synk::Barrier::getInstance()->wait(tid);

      if (0 == tid) {
        times[iter] = omp_get_wtime() - t;

        if (iter == REPEAT - 1) {
          printf("MKL_CSRMM_parbatch: ");
          printEfficiency(times, REPEAT, flop*NBATCH, denseFlop*NBATCH, byte*NBATCH);

          for (int b = 0; b < NBATCH; ++b) {
            correctnessCheck(C_ref[b], C[b], A->m*N, tol);
            memset(C[b], 0, sizeof(float)*A->m*N);
          }
        }
      }
    }
  } // omp parallel

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
            printEfficiency(times, REPEAT*NBATCH, flop, denseFlop, byte);

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
        times[iter] = omp_get_wtime() - t;

        if (iter == REPEAT - 1) {
          printf("myCSRMM_concatenated: ");
          printEfficiency(times, REPEAT, flop*NBATCH, denseFlop*NBATCH, byte*NBATCH);

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
        times[iter] = omp_get_wtime() - t;

        if (iter == REPEAT - 1) {
          printf("myCSRMM_parbatch: ");
          printEfficiency(times, REPEAT, flop*NBATCH, denseFlop*NBATCH, byte*NBATCH);

          for (int b = 0; b < NBATCH; ++b) {
            correctnessCheck(C_ref[b], C[b], A->m*N, tol);
            memset(C[b], 0, sizeof(float)*A->m*N);
          }
        }
      }
    }
  } // omp parallel

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
          printEfficiency(times, REPEAT*NBATCH, flop, denseFlop, byte);

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

      times[iter] = omp_get_wtime() - t;

      if (iter == REPEAT - 1) {
        printf("MKL_CSRMM_reordered_concatenated: ");
        printEfficiency(times, REPEAT, flop*NBATCH, denseFlop*NBATCH, byte*NBATCH);

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
          times[iter] = omp_get_wtime() - t;

          if (iter == REPEAT - 1) {
            printf("MKL_CSRMM_reordered_parbatch: ");
            printEfficiency(times, REPEAT, flop*NBATCH, denseFlop*NBATCH, byte*NBATCH);
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
              printEfficiency(times, REPEAT*NBATCH, flop, denseFlop, byte);

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
          times[iter] = omp_get_wtime() - t;

          if (iter == REPEAT - 1) {
            printf("myCSRMM_reordered_concatenated: ");
            printEfficiency(times, REPEAT, flop*NBATCH, denseFlop*NBATCH, byte*NBATCH);

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
          times[iter] = omp_get_wtime() - t;

          if (iter == REPEAT - 1) {
            printf("myCSRMM_reordered_parbatch: ");
            printEfficiency(times, REPEAT, flop*NBATCH, denseFlop*NBATCH, byte*NBATCH);

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
          printEfficiency(times, REPEAT*NBATCH, flop, denseFlop, byte);

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

      times[iter] = omp_get_wtime() - t;

      if (iter == REPEAT - 1) {
        printf("dense_concatenated: ");
        printEfficiency(times, REPEAT, flop*NBATCH, denseFlop*NBATCH, byte*NBATCH);

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
        times[iter] = omp_get_wtime() - t;

        if (iter == REPEAT - 1) {
          printf("dense_parbatch: ");
          printEfficiency(times, REPEAT, flop*NBATCH, denseFlop*NBATCH, byte*NBATCH);

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

  {
    for (int iter = 0; iter < REPEAT; ++iter) {
      flushLlc();

      for (int b = 0; b < NBATCH; ++b) {
        double t = omp_get_wtime();

        cblas_sgemm(
          CblasRowMajor, CblasNoTrans, CblasNoTrans, 
          A->m, N, nNonZeroCols,
          alpha, A_compressed, nNonZeroCols,
          B_compressed[b], N,
          beta, C[b], N);

        times[iter*NBATCH + b] = omp_get_wtime() - t;

        if (iter == REPEAT - 1 && b == NBATCH - 1) {
          printf("compressed: ");
          printEfficiency(times, REPEAT*NBATCH, flop, denseFlop, byte);

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
        A->m, N_concatenated, nNonZeroCols,
        alpha, A_compressed, nNonZeroCols,
        B_concatenated_compressed, N_concatenated,
        beta, C_concatenated, N_concatenated);

      times[iter] = omp_get_wtime() - t;

      if (iter == REPEAT - 1) {
        printf("compressed_concatenated: ");
        printEfficiency(times, REPEAT, flop*NBATCH, denseFlop*NBATCH, byte*NBATCH);

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
          &A->m, &N, &nNonZeroCols, 
          &alpha, A_compressed, NULL,
          B_compressed[b], NULL,
          &beta, C[b], NULL);
#else
        cblas_sgemm(
          CblasRowMajor, CblasNoTrans, CblasNoTrans, 
          A->m, N, nNonZeroCols,
          alpha, A_compressed, nNonZeroCols,
          B_compressed[b], N,
          beta, C[b], N);
#endif
      }

      synk::Barrier::getInstance()->wait(tid);

      if (0 == tid) {
        times[iter] = omp_get_wtime() - t;

        if (iter == REPEAT - 1) {
          printf("compressed_parbatch: ");
          printEfficiency(times, REPEAT, flop*NBATCH, denseFlop*NBATCH, byte*NBATCH);

          for (int b = 0; b < NBATCH; ++b) {
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
