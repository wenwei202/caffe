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
 *        This example runs symmetric GS smoothing, but SpMP also can
 *        be used for sparse triangular solver, ILU factorization, and so on.
 *
 * \ref "Sparsifying Synchronizations for High-Performance Shared-Memory Sparse
 *      Triangular Solver", Park et al., ISC 2014
 *
 * Expected performance
   (inline_1.mtx can be downloaded from U of Florida matrix collection)
  
 In a 18-core Xeon E5-2699 v3 @ 2.3GHz, 56 gbps STREAM BW

OMP_NUM_THREADS=18 KMP_AFFINITY=granularity=fine,compact,1 test/gs_test 192
input=192
parallelism 5289.901345
fwd_ref                    1.65 gflops   10.00 gbps
bwd_ref                    1.45 gflops    8.83 gbps
fwd_barrier                3.54 gflops   21.48 gbps
bwd_barrier                3.31 gflops   20.11 gbps
fwd_p2p                    4.04 gflops   24.55 gbps
bwd_p2p                    3.66 gflops   22.24 gbps
fwd_p2p_tr_red             4.40 gflops   26.72 gbps
bwd_p2p_tr_red             3.84 gflops   23.33 gbps
fwd_barrier_perm           8.82 gflops   53.59 gbps
bwd_barrier_perm           8.40 gflops   51.05 gbps
fwd_p2p_perm               9.19 gflops   55.82 gbps
bwd_p2p_perm               8.70 gflops   52.85 gbps
fwd_p2p_tr_red_perm        9.21 gflops   55.96 gbps
bwd_p2p_tr_red_perm        8.70 gflops   52.86 gbps

OMP_NUM_THREADS=18 KMP_AFFINITY=granularity=fine,compact,1 test/gs_test inline_1.mtx
input=/home/jpark103/matrices/inline_1.mtx
/home/jpark103/matrices/inline_1.mtx:::symmetric m=503712 n=503712 nnz=36816342
parallelism 287.506849
fwd_ref                    1.83 gflops   11.06 gbps
bwd_ref                    1.30 gflops    7.84 gbps
fwd_barrier                3.76 gflops   22.67 gbps
bwd_barrier                3.70 gflops   22.32 gbps
fwd_p2p                    4.23 gflops   25.47 gbps
bwd_p2p                    4.04 gflops   24.36 gbps
fwd_p2p_tr_red             4.34 gflops   26.18 gbps
bwd_p2p_tr_red             4.20 gflops   25.29 gbps
fwd_barrier_perm           7.34 gflops   44.23 gbps
bwd_barrier_perm           6.98 gflops   42.07 gbps
fwd_p2p_perm               8.36 gflops   50.38 gbps
bwd_p2p_perm               7.45 gflops   44.88 gbps
fwd_p2p_tr_red_perm        9.10 gflops   54.83 gbps
bwd_p2p_tr_red_perm        8.52 gflops   51.36 gbps
 */

#include <cassert>
#include <cstring>
#include <climits>
#include <cfloat>

#include <omp.h>

#include "../LevelSchedule.hpp"
#include "../synk/barrier.hpp"

#include "test.hpp"

/**
 * Reference sequential Gauss-Seidel smoother
 */
template<bool IS_FORWARD>
void gsRef_(const CSR& A, double y[], const double b[])
{
  ADJUST_FOR_BASE;

  int iBegin = IS_FORWARD ? 0 : A.m - 1;
  int iEnd = IS_FORWARD ? A.m : -1;
  int iDelta = IS_FORWARD ? 1 : -1;

  for (int i = iBegin + base; i != iEnd + base; i += iDelta) {
    double sum = b[i];
    for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
      sum -= values[j]*y[colidx[j]];
    }
    y[i] += sum*idiag[i];
  } // for each row
}

void forwardGSRef(const CSR& A, double y[], const double b[])
{
  gsRef_<true>(A, y, b);
}

void backwardGSRef(const CSR& A, double y[], const double b[])
{
  gsRef_<false>(A, y, b);
}

/**
 * Forward Gauss-Seidel smoother parallelized with level scheduling
 * and barrier synchronization
 */
void forwardGSWithBarrier(
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
        y[row] += sum*idiag[row];
      } // for each row

      synk::Barrier::getInstance()->wait(tid);
    } // for each level
  } // omp parallel
}

/**
 * Backward Gauss-Seidel smoother parallelized with level scheduling
 * and barrier synchronization
 */
void backwardGSWithBarrier(
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
        y[row] += sum*idiag[row];
      } // for each row
      synk::Barrier::getInstance()->wait(tid);
    } // for each level
  } // omp parallel
}

/**
 * Forward Gauss-Seidel smoother parallelized with level scheduling
 * and point-to-point synchronization
 */
void forwardGS(
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
        y[row] += sum*idiag[row];
      }

      SPMP_LEVEL_SCHEDULE_NOTIFY;
    } // for each task
  } // omp parallel
}

/**
 * Backward Gauss-Seidel smoother parallelized with level scheduling
 * and point-to-point synchronization
 */
void backwardGS(
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
        y[row] += sum*idiag[row];
      }

      SPMP_LEVEL_SCHEDULE_NOTIFY;
    } // for each task
  } // omp parallel
}

/**
 * Forward Gauss-Seidel smoother parallelized with level scheduling
 * and barrier synchronization. Matrix is reordered.
 */
void forwardGSWithBarrierAndReorderedMatrix(
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
        y[i] += sum*idiag[i];
      } // for each row
      synk::Barrier::getInstance()->wait(tid);
    } // for each level
  } // omp parallel
}

/**
 * Backward Gauss-Seidel smoother parallelized with level scheduling
 * and barrier synchronization. Matrix is reordered.
 */
void backwardGSWithBarrierAndReorderedMatrix(
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
        y[i] += sum*idiag[i];
      } // for each row
      synk::Barrier::getInstance()->wait(tid);
    } // for each level
  } // omp parallel
}

/**
 * Forward Gauss-Seidel smoother parallelized with level scheduling
 * and point-to-point synchronization
 */
void forwardGSWithReorderedMatrix(
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
        y[i] += sum*idiag[i];
      }

      SPMP_LEVEL_SCHEDULE_NOTIFY;
    } // for each task
  } // omp parallel
}

/**
 * Backward Gauss-Seidel smoother parallelized with level scheduling
 * and point-to-point synchronization
 */
void backwardGSWithReorderedMatrix(
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
        y[i] += sum*idiag[i];
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

  CSR *APerm = A->permute(perm, invPerm, true /*sort*/);

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
    flop = 2*nnz;
    byte = nnz*(sizeof(double) + sizeof(int)) + A->m*sizeof(int);
  }

  // allocate a large buffer to flush out cache
  bufToFlushLlc = (double *)_mm_malloc(LLC_CAPACITY, 64);

  /////////////////////////////////////////////////////////////////////////////
  // GS smoother w/o reordering
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
        forwardGSRef(*A, y, b); break;
      case BARRIER :
        forwardGSWithBarrier(*A, y, b, *barrierSchedule, invPerm); break;
      case P2P :
        forwardGS(*A, y, b, *p2pSchedule, invPerm); break;
      case P2P_WITH_TRANSITIVE_REDUCTION :
        forwardGS(*A, y, b, *p2pScheduleWithTransitiveReduction, invPerm); break;
      default: assert(false); break;
      }

      timesForward[i] = omp_get_wtime() - t;
      t = omp_get_wtime();

      switch (option) {
      case REFERENCE :
        backwardGSRef(*A, x, y); break;
      case BARRIER :
        backwardGSWithBarrier(*A, x, y, *barrierSchedule, invPerm); break;
      case P2P :
        backwardGS(*A, x, y, *p2pSchedule, invPerm); break;
      case P2P_WITH_TRANSITIVE_REDUCTION :
        backwardGS(*A, x, y, *p2pScheduleWithTransitiveReduction, invPerm); break;
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
  // GS smoother w/ reordering
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
        forwardGSWithBarrierAndReorderedMatrix(
          *APerm, y, bPerm, *barrierSchedule);
        break;
      case P2P :
        forwardGSWithReorderedMatrix(
          *APerm, y, bPerm, *p2pSchedule);
        break;
      case P2P_WITH_TRANSITIVE_REDUCTION :
        forwardGSWithReorderedMatrix(
          *APerm, y, bPerm, *p2pScheduleWithTransitiveReduction);
        break;
      default: assert(false); break;
      }

      timesForward[i] = omp_get_wtime() - t;
      t = omp_get_wtime();

      switch (option) {
      case BARRIER :
        backwardGSWithBarrierAndReorderedMatrix(
          *APerm, x, y, *barrierSchedule);
        break;
      case P2P :
        backwardGSWithReorderedMatrix(
          *APerm, x, y, *p2pSchedule);
        break;
      case P2P_WITH_TRANSITIVE_REDUCTION :
        backwardGSWithReorderedMatrix(
          *APerm, x, y, *p2pScheduleWithTransitiveReduction);
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

  delete barrierSchedule;
  delete p2pSchedule;
  delete p2pScheduleWithTransitiveReduction;

  delete A;
  delete APerm;

  FREE(b);
  FREE(y);
  FREE(bPerm);
  FREE(tempVector);

  synk::Barrier::deleteInstance();

  return 0;
}
