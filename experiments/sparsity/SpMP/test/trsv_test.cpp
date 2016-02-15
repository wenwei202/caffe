/**
Copyright (c) 2015, Intel Corporation. All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Intel Corporation nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL INTEL CORPORATION BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*!
 * \brief Example of parallelizing GS-like loops with data-dependent
 *        loop carried dependencies.
 *        This example runs symmetric GS preconditioner, but SpMP also can
 *        be used for ILU factorization and so on.
 *
 * \ref "Sparsifying Synchronizations for High-Performance Shared-Memory Sparse
 *      Triangular Solver", Park et al., ISC 2014
 *
 * Expected performance
   (inline_1.mtx can be downloaded from U of Florida matrix collection)
  
 In a 18-core Xeon E5-2699 v3 @ 2.3GHz, 56 gbps STREAM BW

OMP_NUM_THREADS=18 KMP_AFFINITY=granularity=fine,compact,1 test/trsv_test 192
input=192
parallelism 5289.901345
fwd_ref                    1.41 gflops    8.64 gbps
bwd_ref                    1.32 gflops    8.12 gbps
fwd_barrier                3.80 gflops   23.33 gbps
bwd_barrier                3.56 gflops   21.86 gbps
fwd_p2p                    3.67 gflops   22.57 gbps
bwd_p2p                    3.46 gflops   21.23 gbps
fwd_p2p_tr_red             3.68 gflops   22.60 gbps
bwd_p2p_tr_red             3.46 gflops   21.25 gbps
fwd_barrier_perm           8.33 gflops   51.17 gbps
bwd_barrier_perm           8.00 gflops   49.13 gbps
fwd_p2p_perm               8.73 gflops   53.63 gbps
bwd_p2p_perm               8.43 gflops   51.77 gbps
fwd_p2p_tr_red_perm        8.72 gflops   53.56 gbps
bwd_p2p_tr_red_perm        8.41 gflops   51.69 gbps

OMP_NUM_THREADS=18 KMP_AFFINITY=granularity=fine,compact,1 test/trsv_test inline_1.mtx
input=/home/jpark103/matrices/inline_1.mtx
/home/jpark103/matrices/inline_1.mtx:::symmetric m=503712 n=503712 nnz=36816342
parallelism 287.506849
fwd_ref                    1.63 gflops    9.89 gbps
bwd_ref                    1.23 gflops    7.42 gbps
fwd_barrier                3.55 gflops   21.48 gbps
bwd_barrier                3.04 gflops   18.40 gbps
fwd_p2p                    3.89 gflops   23.55 gbps
bwd_p2p                    3.94 gflops   23.86 gbps
fwd_p2p_tr_red             4.19 gflops   25.37 gbps
bwd_p2p_tr_red             4.46 gflops   27.00 gbps
fwd_barrier_perm           5.86 gflops   35.49 gbps
bwd_barrier_perm           5.51 gflops   33.38 gbps
fwd_p2p_perm               6.91 gflops   41.86 gbps
bwd_p2p_perm               5.82 gflops   35.26 gbps
fwd_p2p_tr_red_perm        8.01 gflops   48.49 gbps
bwd_p2p_tr_red_perm        7.22 gflops   43.74 gbps
 */

#include <cassert>
#include <cstring>
#include <climits>
#include <cfloat>

#include <omp.h>

#ifdef MKL
#include <mkl.h>
#endif

#include "../LevelSchedule.hpp"
#include "../synk/barrier.hpp"

#include "test.hpp"

/**
 * Reference sequential sparse triangular solver
 */
void forwardSolveRef(const CSR& A, double y[], const double b[])
{
  ADJUST_FOR_BASE;

  for (int i = base; i < A.m + base; ++i) {
    double sum = b[i];
    for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
      sum -= values[j]*y[colidx[j]];
    }
    y[i] = sum*idiag[i];
  } // for each row
}

void backwardSolveRef(const CSR& A, double y[], const double b[])
{
  ADJUST_FOR_BASE;

  for (int i = A.m - 1 + base; i >= base; --i) {
    double sum = b[i];
    for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
      sum -= values[j]*y[colidx[j]];
    }
    y[i] = sum;
  } // for each row
}

/**
 * Forward sparse triangular solver parallelized with level scheduling
 * and barrier synchronization
 */
void forwardSolveWithBarrier(
  const CSR& A, double y[], const double b[],
  const LevelSchedule& schedule,
  const int *perm)
{
  ADJUST_FOR_BASE;

#pragma omp parallel
  {
    int tid = omp_get_thread_num();

    const vector<int>& threadBoundaries = schedule.threadBoundaries;
    const vector<int>& taskBoundaries = schedule.taskBoundaries;

    for (int task = threadBoundaries[tid]; task < threadBoundaries[tid + 1]; ++task) {
      for (int i = taskBoundaries[task]; i < taskBoundaries[task + 1]; ++i) {
        int row = perm[i] + base;
        double sum = b[row];
        for (int j = rowptr[row]; j < rowptr[row + 1]; ++j) {
          sum -= values[j]*y[colidx[j]];
        }
        y[row] = sum*idiag[row];
      } // for each row

      synk::Barrier::getInstance()->wait(tid);
    } // for each level
  } // omp parallel
}

/**
 * Backward sparse triangular solver parallelized with level scheduling
 * and barrier synchronization
 */
void backwardSolveWithBarrier(
  const CSR& A, double y[], const double b[],
  const LevelSchedule& schedule,
  const int *perm)
{
  ADJUST_FOR_BASE;

#pragma omp parallel
  {
    int tid = omp_get_thread_num();

    const vector<int>& threadBoundaries = schedule.threadBoundaries;
    const vector<int>& taskBoundaries = schedule.taskBoundaries;

    for (int task = threadBoundaries[tid + 1] - 1; task >= threadBoundaries[tid]; --task) {
      for (int i = taskBoundaries[task + 1] - 1; i >= taskBoundaries[task]; --i) {
        int row = perm[i] + base;
        double sum = b[row];
        for (int j = rowptr[row + 1] - 1; j >= rowptr[row]; --j) {
          sum -= values[j]*y[colidx[j]];
        }
        y[row] = sum;
      } // for each row
      synk::Barrier::getInstance()->wait(tid);
    } // for each level
  } // omp parallel
}

/**
 * Forward sparse triangular solver parallelized with level scheduling
 * and point-to-point synchronization
 */
void forwardSolve(
  const CSR& A, double y[], const double b[],
  const LevelSchedule& schedule,
  const int *perm)
{
  ADJUST_FOR_BASE;

#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    const int ntasks = schedule.ntasks;
    const short *nparents = schedule.nparentsForward;
    const vector<int>& threadBoundaries = schedule.threadBoundaries;
    const vector<int>& taskBoundaries = schedule.taskBoundaries;

    int nPerThread = (ntasks + nthreads - 1)/nthreads;
    int nBegin = min(nPerThread*tid, ntasks);
    int nEnd = min(nBegin + nPerThread, ntasks);

    volatile int *taskFinished = schedule.taskFinished;
    int **parents = schedule.parentsForward;

    memset((char *)(taskFinished + nBegin), 0, (nEnd - nBegin)*sizeof(int));

    synk::Barrier::getInstance()->wait(tid);

    for (int task = threadBoundaries[tid]; task < threadBoundaries[tid + 1]; ++task) {
      SPMP_LEVEL_SCHEDULE_WAIT;

      for (int i = taskBoundaries[task]; i < taskBoundaries[task + 1]; ++i) {
        int row = perm[i] + base;
        double sum = b[row];
        for (int j = rowptr[row]; j < rowptr[row + 1]; ++j) {
          sum -= values[j]*y[colidx[j]];
        }
        y[row] = sum*idiag[row];
      }

      SPMP_LEVEL_SCHEDULE_NOTIFY;
    } // for each task
  } // omp parallel
}

/**
 * Backward sparse triangular solver parallelized with level scheduling
 * and point-to-point synchronization
 */
void backwardSolve(
  const CSR& A, double y[], const double b[],
  const LevelSchedule& schedule,
  const int *perm)
{
  ADJUST_FOR_BASE;

#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    const int ntasks = schedule.ntasks;
    const short *nparents = schedule.nparentsBackward;
    const vector<int>& threadBoundaries = schedule.threadBoundaries;
    const vector<int>& taskBoundaries = schedule.taskBoundaries;

    int nPerThread = (ntasks + nthreads - 1)/nthreads;
    int nBegin = min(nPerThread*tid, ntasks);
    int nEnd = min(nBegin + nPerThread, ntasks);

    volatile int *taskFinished = schedule.taskFinished;
    int **parents = schedule.parentsBackward;

    memset((char *)(taskFinished + nBegin), 0, (nEnd - nBegin)*sizeof(int));

    synk::Barrier::getInstance()->wait(tid);

    for (int task = threadBoundaries[tid + 1] - 1; task >= threadBoundaries[tid]; --task) {
      SPMP_LEVEL_SCHEDULE_WAIT;

      for (int i = taskBoundaries[task + 1] - 1; i >= taskBoundaries[task]; --i) {
        int row = perm[i] + base;
        double sum = b[row];
        for (int j = rowptr[row + 1] - 1; j >= rowptr[row]; --j) {
          sum -= values[j]*y[colidx[j]];
        }
        y[row] = sum;
      }

      SPMP_LEVEL_SCHEDULE_NOTIFY;
    } // for each task
  } // omp parallel
}

/**
 * Forward sparse triangular solver parallelized with level scheduling
 * and barrier synchronization. Matrix is reordered.
 */
void forwardSolveWithBarrierAndReorderedMatrix(
  const CSR& A, double y[], const double b[],
  const LevelSchedule& schedule)
{
  ADJUST_FOR_BASE;

#pragma omp parallel
  {
    int tid = omp_get_thread_num();

    const vector<int>& threadBoundaries = schedule.threadBoundaries;
    const vector<int>& taskBoundaries = schedule.taskBoundaries;

    for (int task = threadBoundaries[tid]; task < threadBoundaries[tid + 1]; ++task) {
      for (int i = taskBoundaries[task] + base; i < taskBoundaries[task + 1] + base; ++i) {
        double sum = b[i];
        for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
          sum -= values[j]*y[colidx[j]];
        }
        y[i] = sum*idiag[i];
      } // for each row
      synk::Barrier::getInstance()->wait(tid);
    } // for each level
  } // omp parallel
}

/**
 * Backward sparse triangular solver parallelized with level scheduling
 * and barrier synchronization. Matrix is reordered.
 */
void backwardSolveWithBarrierAndReorderedMatrix(
  const CSR& A, double y[], const double b[],
  const LevelSchedule& schedule)
{
  ADJUST_FOR_BASE;

#pragma omp parallel
  {
    int tid = omp_get_thread_num();

    const vector<int>& threadBoundaries = schedule.threadBoundaries;
    const vector<int>& taskBoundaries = schedule.taskBoundaries;

    for (int task = threadBoundaries[tid + 1] - 1; task >= threadBoundaries[tid]; --task) {
      for (int i = taskBoundaries[task + 1] - 1 + base; i >= taskBoundaries[task] + base; --i) {
        double sum = b[i];
        for (int j = rowptr[i + 1] - 1; j >= rowptr[i]; --j) {
          sum -= values[j]*y[colidx[j]];
        }
        y[i] = sum;
      } // for each row
      synk::Barrier::getInstance()->wait(tid);
    } // for each level
  } // omp parallel
}

/**
 * Forward sparse triangular solver parallelized with level scheduling
 * and point-to-point synchronization
 */
void forwardSolveWithReorderedMatrix(
  const CSR& A, double y[], const double b[],
  const LevelSchedule& schedule)
{
  ADJUST_FOR_BASE;

#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    const int ntasks = schedule.ntasks;
    const short *nparents = schedule.nparentsForward;
    const vector<int>& threadBoundaries = schedule.threadBoundaries;
    const vector<int>& taskBoundaries = schedule.taskBoundaries;

    int nPerThread = (ntasks + nthreads - 1)/nthreads;
    int nBegin = min(nPerThread*tid, ntasks);
    int nEnd = min(nBegin + nPerThread, ntasks);

    volatile int *taskFinished = schedule.taskFinished;
    int **parents = schedule.parentsForward;

    memset((char *)(taskFinished + nBegin), 0, (nEnd - nBegin)*sizeof(int));

    synk::Barrier::getInstance()->wait(tid);

    for (int task = threadBoundaries[tid]; task < threadBoundaries[tid + 1]; ++task) {
      SPMP_LEVEL_SCHEDULE_WAIT;

      for (int i = taskBoundaries[task] + base; i < taskBoundaries[task + 1] + base; ++i) {
        double sum = b[i];
        for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
          sum -= values[j]*y[colidx[j]];
        }
        y[i] = sum*idiag[i];
      }

      SPMP_LEVEL_SCHEDULE_NOTIFY;
    } // for each task
  } // omp parallel
}

/**
 * Backward sparse triangular solver parallelized with level scheduling
 * and point-to-point synchronization
 */
void backwardSolveWithReorderedMatrix(
  const CSR& A, double y[], const double b[],
  const LevelSchedule& schedule)
{
  ADJUST_FOR_BASE;

#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    const int ntasks = schedule.ntasks;
    const short *nparents = schedule.nparentsBackward;
    const vector<int>& threadBoundaries = schedule.threadBoundaries;
    const vector<int>& taskBoundaries = schedule.taskBoundaries;

    int nPerThread = (ntasks + nthreads - 1)/nthreads;
    int nBegin = min(nPerThread*tid, ntasks);
    int nEnd = min(nBegin + nPerThread, ntasks);

    volatile int *taskFinished = schedule.taskFinished;
    int **parents = schedule.parentsBackward;

    memset((char *)(taskFinished + nBegin), 0, (nEnd - nBegin)*sizeof(int));

    synk::Barrier::getInstance()->wait(tid);

    for (int task = threadBoundaries[tid + 1] - 1; task >= threadBoundaries[tid]; --task) {
      SPMP_LEVEL_SCHEDULE_WAIT;

      for (int i = taskBoundaries[task + 1] - 1 + base; i >= taskBoundaries[task] + base; --i) {
        double sum = b[i];
        for (int j = rowptr[i + 1] - 1; j >= rowptr[i]; --j) {
          sum -= values[j]*y[colidx[j]];
        }
        y[i] = sum;
      }

      SPMP_LEVEL_SCHEDULE_NOTIFY;
    } // for each task
  } // omp parallel
}

int main(int argc, char **argv)
{
  double tBegin = omp_get_wtime();

  /////////////////////////////////////////////////////////////////////////////
  // Load input
  /////////////////////////////////////////////////////////////////////////////

  int m = argc > 1 ? atoi(argv[1]) : 64; // default input is 64^3 27-pt 3D Lap.
  if (argc < 2) {
    fprintf(
      stderr,
      "Using default 64^3 27-pt 3D Laplacian matrix\n"
      "-- Usage examples --\n"
      "  %s 128 : 128^3 27-pt 3D Laplacian matrix\n"
      "  %s inline_1.mtx: run with inline_1 matrix in matrix market format\n\n",
      argv[0], argv[0]);
  }
  char buf[1024];
  sprintf(buf, "%d", m);

  bool readFromFile = argc > 1 ? strcmp(buf, argv[1]) && !strstr(argv[1], ".mtx"): false;
  printf("input=%s\n", argc > 1 ? argv[1] : buf);

  CSR *A = new CSR(argc > 1 ? argv[1] : buf);

  /////////////////////////////////////////////////////////////////////////////
  // Split lower and upper triangular parts
  /////////////////////////////////////////////////////////////////////////////

  CSR *L = new CSR, *U = new CSR;
  splitLU(*A, L, U);

  /////////////////////////////////////////////////////////////////////////////
  // Construct schedules
  /////////////////////////////////////////////////////////////////////////////

  LevelSchedule *barrierSchedule = new LevelSchedule;
  barrierSchedule->useBarrier = true;
  barrierSchedule->transitiveReduction = false;
  barrierSchedule->constructTaskGraph(*A);

  LevelSchedule *p2pSchedule = new LevelSchedule;
  p2pSchedule->transitiveReduction = false;
  p2pSchedule->constructTaskGraph(*A);

  LevelSchedule *p2pScheduleWithTransitiveReduction = new LevelSchedule;
  p2pScheduleWithTransitiveReduction->constructTaskGraph(*A);

  printf("parallelism %f\n", (double)A->m/(barrierSchedule->levIndices.size() - 1));
  assert(barrierSchedule->levIndices.size() == p2pSchedule->levIndices.size());
  assert(barrierSchedule->levIndices.size() == p2pScheduleWithTransitiveReduction->levIndices.size());

  /////////////////////////////////////////////////////////////////////////////
  // Reorder matrix
  /////////////////////////////////////////////////////////////////////////////

  const int *perm = p2pScheduleWithTransitiveReduction->origToThreadContPerm;
  const int *invPerm = p2pScheduleWithTransitiveReduction->threadContToOrigPerm;
  assert(isPerm(perm, A->m));
  assert(isPerm(invPerm, A->m));

  CSR *LPerm = L->permute(perm, invPerm);
  CSR *UPerm = U->permute(perm, invPerm);

  /////////////////////////////////////////////////////////////////////////////
  // Allocate vectors
  /////////////////////////////////////////////////////////////////////////////

  double *b = MALLOC(double, A->m);
#pragma omp parallel for
  for(int i=0; i < A->m; i++) b[i] = i;

  double *y = MALLOC(double, A->m);
  double *x = MALLOC(double, A->m);

  double flop, byte;
  for (int i = 0; i < 2; ++i) {
    int nnz = A->rowptr[A->m];
    flop = 2*((nnz - A->m)/2 + A->m);
    byte = ((nnz - A->m)/2 + A->m)*(sizeof(double) + sizeof(int)) + A->m*sizeof(int);
  }

  // allocate a large buffer to flush out cache
  bufToFlushLlc = (double *)_mm_malloc(LLC_CAPACITY, 64);

  /////////////////////////////////////////////////////////////////////////////
  // sparse triangular solver w/o reordering
  /////////////////////////////////////////////////////////////////////////////

  int REPEAT = 128;
  double timesForward[REPEAT], timesBackward[REPEAT];

  for (int o = REFERENCE; o <= P2P_WITH_TRANSITIVE_REDUCTION; ++o) {
    SynchronizationOption option = (SynchronizationOption)o;

    for (int i = 0; i < REPEAT; ++i) {
      flushLlc();

      initializeX(x, A->m);
      initializeX(y, A->m);

      double t = omp_get_wtime();

      switch (option) {
      case REFERENCE :
        forwardSolveRef(*L, y, b); break;
      case BARRIER :
        forwardSolveWithBarrier(*L, y, b, *barrierSchedule, invPerm); break;
      case P2P :
        forwardSolve(*L, y, b, *p2pSchedule, invPerm); break;
      case P2P_WITH_TRANSITIVE_REDUCTION :
        forwardSolve(*L, y, b, *p2pScheduleWithTransitiveReduction, invPerm); break;
      default: assert(false); break;
      }

      timesForward[i] = omp_get_wtime() - t;
      t = omp_get_wtime();

      switch (option) {
      case REFERENCE :
        backwardSolveRef(*U, x, y); break;
      case BARRIER :
        backwardSolveWithBarrier(*U, x, y, *barrierSchedule, invPerm); break;
      case P2P :
        backwardSolve(*U, x, y, *p2pSchedule, invPerm); break;
      case P2P_WITH_TRANSITIVE_REDUCTION :
        backwardSolve(*U, x, y, *p2pScheduleWithTransitiveReduction, invPerm); break;
      default: assert(false); break;
      }

      timesBackward[i] = omp_get_wtime() - t;

      if (i == REPEAT - 1) {
        for (int j = 0; j < 2; ++j) {
          printf(0 == j ? "fwd_" : "bwd_");
          switch (option) {
          case REFERENCE : printf("ref\t\t\t"); break;
          case BARRIER: printf("barrier\t\t"); break;
          case P2P: printf("p2p\t\t\t"); break;
          case P2P_WITH_TRANSITIVE_REDUCTION: printf("p2p_tr_red\t\t"); break;
          default: assert(false); break;
          }
          printEfficiency(
            0 == j ? timesForward : timesBackward, REPEAT, flop, byte);
        }

        correctnessCheck(A, x);
      }
    } // for each iteration
  } // for each option

  /////////////////////////////////////////////////////////////////////////////
  // sparse triangular solver w/ reordering
  /////////////////////////////////////////////////////////////////////////////

  double *bPerm = getReorderVector(b, perm, A->m);
  double *tempVector = MALLOC(double, A->m);

  for (int o = BARRIER; o <= P2P_WITH_TRANSITIVE_REDUCTION; ++o) {
    SynchronizationOption option = (SynchronizationOption)o;

    for (int i = 0; i < REPEAT; ++i) {
      flushLlc();

      initializeX(x, A->m);
      initializeX(y, A->m);
      reorderVector(x, tempVector, perm, A->m);
      reorderVector(y, tempVector, perm, A->m);

      double t = omp_get_wtime();

      switch (option) {
      case BARRIER :
        forwardSolveWithBarrierAndReorderedMatrix(
          *LPerm, y, bPerm, *barrierSchedule);
        break;
      case P2P :
        forwardSolveWithReorderedMatrix(
          *LPerm, y, bPerm, *p2pSchedule);
        break;
      case P2P_WITH_TRANSITIVE_REDUCTION :
        forwardSolveWithReorderedMatrix(
          *LPerm, y, bPerm, *p2pScheduleWithTransitiveReduction);
        break;
      default: assert(false); break;
      }

      timesForward[i] = omp_get_wtime() - t;
      t = omp_get_wtime();

      switch (option) {
      case BARRIER :
        backwardSolveWithBarrierAndReorderedMatrix(
          *UPerm, x, y, *barrierSchedule);
        break;
      case P2P :
        backwardSolveWithReorderedMatrix(
          *UPerm, x, y, *p2pSchedule);
        break;
      case P2P_WITH_TRANSITIVE_REDUCTION :
        backwardSolveWithReorderedMatrix(
          *UPerm, x, y, *p2pScheduleWithTransitiveReduction);
        break;
      default: assert(false); break;
      }

      timesBackward[i] = omp_get_wtime() - t;

      if (i == REPEAT - 1) {
        for (int j = 0; j < 2; ++j) {
          printf(0 == j ? "fwd_" : "bwd_");
          switch (option) {
          case BARRIER: printf("barrier_perm\t"); break;
          case P2P: printf("p2p_perm\t\t"); break;
          case P2P_WITH_TRANSITIVE_REDUCTION: printf("p2p_tr_red_perm\t"); break;
          default: assert(false); break;
          }
          printEfficiency(
            0 == j ? timesForward : timesBackward, REPEAT, flop, byte);
        }

        reorderVector(x, tempVector, invPerm, A->m);
        correctnessCheck(A, x);
      }
    }
  }

#ifdef MKL

  /////////////////////////////////////////////////////////////////////////////
  // Inspector-Executor interface in MKL 11.3+
  /////////////////////////////////////////////////////////////////////////////

  sparse_matrix_t mklA;
  sparse_status_t stat = mkl_sparse_d_create_csr(
    &mklA,
    SPARSE_INDEX_BASE_ZERO, A->m, A->n,
    A->rowptr, A->rowptr + 1,
    A->colidx, A->values);

  if (SPARSE_STATUS_SUCCESS != stat) {
    fprintf(stderr, "Failed to create mkl csr\n");
    return -1;
  }

  matrix_descr descL;
  descL.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  descL.mode = SPARSE_FILL_MODE_LOWER;
  descL.diag = SPARSE_DIAG_NON_UNIT;

  sparse_matrix_t mklL, mklU;
  stat = mkl_sparse_copy(mklA, descL, &mklL);
  if (SPARSE_STATUS_SUCCESS != stat) {
    fprintf(stderr, "Failed to create mkl csr lower\n");
    return -1;
  }

  stat = mkl_sparse_set_sv_hint(
    mklL, SPARSE_OPERATION_NON_TRANSPOSE, descL, REPEAT);

  if (SPARSE_STATUS_SUCCESS != stat) {
    fprintf(stderr, "Failed to set sv hint\n");
    return -1;
  }

  stat = mkl_sparse_optimize(mklL);

  if (SPARSE_STATUS_SUCCESS != stat) {
    fprintf(stderr, "Failed to sparse optimize\n");
    return -1;
  }

  matrix_descr descU;
  descU.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  descU.mode = SPARSE_FILL_MODE_UPPER;
  descU.diag = SPARSE_DIAG_UNIT;

  stat = mkl_sparse_copy(mklA, descU, &mklU);
  if (SPARSE_STATUS_SUCCESS != stat) {
    fprintf(stderr, "Failed to create mkl csr upper\n");
    return -1;
  }

  stat = mkl_sparse_set_sv_hint(
    mklU, SPARSE_OPERATION_NON_TRANSPOSE, descU, REPEAT);

  if (SPARSE_STATUS_SUCCESS != stat) {
    fprintf(stderr, "Failed to set sv hint\n");
    return -1;
  }

  stat = mkl_sparse_optimize(mklU);

  if (SPARSE_STATUS_SUCCESS != stat) {
    fprintf(stderr, "Failed to sparse optimize\n");
    return -1;
  }

  for (int i = 0; i < REPEAT; ++i) {
    flushLlc();

    initializeX(x, A->m);
    initializeX(y, A->m);

    double t = omp_get_wtime();

    mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1, mklL, descL, b, y);

    timesForward[i] = omp_get_wtime() - t;
    t = omp_get_wtime();

    mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE, 1, mklU, descU, y, x);

    timesBackward[i] = omp_get_wtime() - t;

    if (i == REPEAT - 1) {
      for (int j = 0; j < 2; ++j) {
        printf(0 == j ? "fwd_" : "bwd_");
        printf("mkl\t\t\t");
        printEfficiency(
          0 == j ? timesForward : timesBackward, REPEAT, flop, byte);
      }

      correctnessCheck(A, x);
    }
  }

  stat = mkl_sparse_destroy(mklA);
  if (SPARSE_STATUS_SUCCESS != stat) {
    fprintf(stderr, "Failed to destroy mkl csr\n");
    return -1;
  }

#endif

  delete barrierSchedule;
  delete p2pSchedule;
  delete p2pScheduleWithTransitiveReduction;

  delete A;
  delete L;
  delete U;

  delete LPerm;
  delete UPerm;

  FREE(b);
  FREE(y);
  FREE(bPerm);
  FREE(tempVector);

  synk::Barrier::deleteInstance();

  return 0;
}
