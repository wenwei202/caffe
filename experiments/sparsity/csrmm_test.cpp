#include <stdlib.h>
#include <string.h>

#include <omp.h>
#include <mkl.h>

#include "SpMP/CSR.hpp"
#include "SpMP/reordering/BFSBipartite.hpp"
#include "SpMP/test/test.hpp"
#include "SpMP/synk/barrier.hpp"

static void printEfficiency(
  double *times, int REPEAT, double flop, double denseFlop, double byte)
{
  sort(times, times + REPEAT);

  double t = times[REPEAT/2];

  printf(
    "%7.2f gflops (dense: %7.2f gflops) %7.2f gbps\n",
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
  float *C, int ldaC)
{
//#define BLOCK (65536)

  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();

  int total_work = A_rowptr[M];
  int work_per_thread = (total_work + nthreads - 1)/nthreads;

  int begin = tid == 0 ? 0 : std::lower_bound(A_rowptr, A_rowptr + M, work_per_thread*tid) - A_rowptr;
  int end = tid == nthreads - 1 ? M : std::lower_bound(A_rowptr, A_rowptr + M, work_per_thread*(tid + 1)) - A_rowptr;

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
  printf("%s: %dx%d %d nnz (%g sparsity)\n", argv[1], A->m, A->n, A->getNnz(), (double)A->getNnz()/(A->m*A->n));

  // A has DP values, create a SP buffer and copy to it
  float *A_values;
  posix_memalign((void **)&A_values, 4096, sizeof(float)*A->getNnz());
  for (int i = 0; i < A->getNnz(); ++i) {
    A_values[i] = A->values[i];
  }

  // Create dense version of A
  float *A_dense;
  posix_memalign((void **)&A_dense, 4096, sizeof(float)*A->m*A->n);
  memset(A_dense, 0, sizeof(float)*A->m*A->n);
  for (int i = 0; i < A->m; ++i) {
    for (int j = A->rowptr[i]; j < A->rowptr[i + 1]; ++j) {
      A_dense[i*A->n + A->colidx[j]] = A_values[j];
    }
  }

  // Initialize B and C
  float *B, *C;
  posix_memalign((void **)&B, 4096, sizeof(float)*A->n*N);
  posix_memalign((void **)&C, 4096, sizeof(float)*A->m*N);

  float *C_ref;
  posix_memalign((void **)&C_ref, 4096, sizeof(float)*A->m*N);

  srand(0); // determinimistic randomization
  for (int i = 0; i < A->n*N; ++i) {
    B[i] = rand();
  }

  double flop = 2*A->getNnz()*N;
  double denseFlop = 2*A->m*A->n*N;
  double byte = (sizeof(float) + sizeof(int))*A->getNnz() + sizeof(int)*A->m + sizeof(float)*(A->n + A->m)*N;

  const int REPEAT = 16;
#ifdef NDEBUG
  double tol = 1e-8;
#else
  double tol = 1e-1; // when compiled with -O0 option, FMA is not used so less accurate
#endif

  const char *matdescra = "GXXCX";
  const char transa = 'N';
  float alpha = 1, beta = 0;

  // 1. Test MKL csrmm
  {
    double timesMKL[REPEAT];

    for (int i = 0; i < REPEAT; ++i) {
      flushLlc();

      double t = omp_get_wtime();

      mkl_scsrmm(
        &transa, &A->m, &N, &A->n,
        &alpha, matdescra,
        A_values, A->colidx, A->rowptr, A->rowptr + 1,
        B, &N,
        &beta, C, &N);

      timesMKL[i] = omp_get_wtime() - t;

      if (i == REPEAT - 1) {
        printf("MKL_CSRMM: ");
        printEfficiency(timesMKL, REPEAT, flop, denseFlop, byte);

        // Copy reference output
        for (int j = 0; j < A->m*N; ++j) {
          C_ref[j] = C[j];
        }
      }
    }
  }

  // 2. Test our own CSRMM
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    double timesOurs[REPEAT];

    for (int i = 0; i < REPEAT; ++i) {
      if (0 == tid) flushLlc();

      double t = omp_get_wtime();

      synk::Barrier::getInstance()->wait(tid);

      my_scsrmm(A->m, N, A->n,
        A_values, A->colidx, A->rowptr,
        B, N,
        C, N);

      synk::Barrier::getInstance()->wait(tid);

      if (0 == tid) {
        timesOurs[i] = omp_get_wtime() - t;

        if (i == REPEAT - 1) {
          printf("myCSRMM: ");
          printEfficiency(timesOurs, REPEAT, flop, denseFlop, byte);

          correctnessCheck(C_ref, C, A->m*N, tol);
        }
      }
    }
  } // omp parallel

  // 3. Test our own CSRMM with reordering
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

    float *B_reordered, *C_reordered;
    posix_memalign((void **)&B_reordered, 4096, sizeof(float)*A->n*N);
    posix_memalign((void **)&C_reordered, 4096, sizeof(float)*A->m*N);

    for (int i = 0; i < A->n; ++i) {
      for (int j = 0; j < N; ++j) {
        B_reordered[colPerm[i]*N + j] = B[i*N + j];
      }
    }

    printf("BW is reduced by BFS reordering: %d -> %d\n", A->getBandwidth(), AReordered->getBandwidth());

    double timesReordered[REPEAT];

    for (int i = 0; i < REPEAT; ++i) {
      flushLlc();

      double t = omp_get_wtime();

      mkl_scsrmm(
        &transa, &A->m, &N, &A->n,
        &alpha, matdescra,
        A_values, AReordered->colidx, AReordered->rowptr, AReordered->rowptr + 1,
        B_reordered, &N,
        &beta, C_reordered, &N);

      timesReordered[i] = omp_get_wtime() - t;

      if (i == REPEAT - 1) {
        printf("MKL_CSRMM_reordered: ");
        printEfficiency(timesReordered, REPEAT, flop, denseFlop, byte);

        for (int i = 0; i < A->m; ++i) {
          for (int j = 0; j < N; ++j) {
            C[rowInversePerm[i]*N + j] = C_reordered[i*N + j];
          }
        }
        correctnessCheck(C_ref, C, AReordered->m*N, tol);
      }
    }

#pragma omp parallel
    {
      int tid = omp_get_thread_num();

      for (int i = 0; i < REPEAT; ++i) {
        if (0 == tid) flushLlc();

        double t = omp_get_wtime();

        synk::Barrier::getInstance()->wait(tid);

        my_scsrmm(AReordered->m, N, AReordered->n,
          A_values, AReordered->colidx, AReordered->rowptr,
          B_reordered, N,
          C_reordered, N);

        synk::Barrier::getInstance()->wait(tid);

        if (0 == tid) {
          timesReordered[i] = omp_get_wtime() - t;

          if (i == REPEAT - 1) {
            printf("myCSRMM_reordered: ");
            printEfficiency(timesReordered, REPEAT, flop, denseFlop, byte);

            for (int i = 0; i < A->m; ++i) {
              for (int j = 0; j < N; ++j) {
                C[rowInversePerm[i]*N + j] = C_reordered[i*N + j];
              }
            }
            correctnessCheck(C_ref, C, AReordered->m*N, tol);
          }
        }
      }
    } // omp parallel

    delete AT;
    delete AReordered;
  }

  // 3. Test dense SGEMM
  {
    double timesDense[REPEAT];

    for (int i = 0; i < REPEAT; ++i) {
      flushLlc();

      double t = omp_get_wtime();

      cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, 
        A->m, N, A->n,
        1, A_dense, A->n,
        B, N,
        0, C, N);

      timesDense[i] = omp_get_wtime() - t;

      if (i == REPEAT - 1) {
        printf("dense: ");
        printEfficiency(timesDense, REPEAT, flop, denseFlop, byte);

        //correctnessCheck(C_ref, C, A->m*N, tol);
      }
    }
  }

  delete A;
  free(A_values);

  free(B);
  free(C);

  return 0;
}
