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
 * \brief Level scheduling of sparse matrix using point-to-point synchronization
 *        with redundant transitive dependency elimination.
 *
 * \author Jongsoo Park (jongsoo.park@intel.com)
 *
 * \ref "Sparsifying Synchronizations for High-Performance Shared-Memory Sparse
 *      Triangular Solver", Park et al., ISC 2014
 */

#pragma once

#include <vector>
#include <map>

#include "CSR.hpp"
#include "Utils.hpp"
#include "MemoryPool.hpp"

namespace SpMP
{

//#define TRACE_TASK_TIMES

//#define MEASURE_PARALLELISM
//#define MEASURE_SPIN_TIME

//#define TRSOLVER_LOG

#define SPMP_LEVEL_SCHEDULE_WAIT_DEFAULT {\
  int n = nparents[task]; \
  int *c = parents[task]; \
  for (int i = 0; i < n; ++i) \
    while (!taskFinished[c[i]]); \
}

#ifdef MEASURE_SPIN_TIME
//#define MEASURE_TASK_TIME
//#define TRACE_SPIN_TIME

extern unsigned long long spin_times[NUM_MAX_THREADS];
#ifdef MEASURE_TASK_TIME
extern unsigned long long task_time_sum[65536];
extern int task_time_cnt[65536];
#endif

#ifdef TRACE_SPIN_TIME
extern unsigned long long *spin_traces[NUM_MAX_THREADS];
extern int spin_trace_counts[NUM_MAX_THREADS];

#define SPMP_LEVEL_SCHEDULE_WAIT {\
  int cnt = spin_trace_counts[tid]; \
  spin_traces[tid][cnt] = __rdtsc(); \
  SPMP_LEVEL_SCHEDULE_WAIT_DEFAULT; \
  spin_traces[tid][cnt + 1] = __rdtsc(); \
  spin_times[tid] += spin_traces[tid][cnt + 1] - spin_traces[tid][cnt]; \
  spin_trace_counts[tid] += 2; \
}

void dumpTrace(const char *fileName);

#elif defined(TRACE_TASK_TIMES)

unsigned long long *taskTimes[NUM_MAX_THREADS];
unsigned long long taskTimeCnt[NUM_MAX_THREADS];

#define SPMP_LEVEL_SCHEDULE_WAIT { \
  taskTimes[tid][taskTimeCnt[tid]] = __rdtsc(); \
  taskTimeCnt[tid]++; \
  SPMP_LEVEL_SCHEDULE_WAIT_DEFAULT; \
  taskTimes[tid][taskTimeCnt[tid]] = __rdtsc(); \
  taskTimeCnt[tid]++; \
}
#else
#define SPMP_LEVEL_SCHEDULE_WAIT { \
  unsigned long long t = __rdtsc(); \
  SPMP_LEVEL_SCHEDULE_WAIT_DEFAULT; \
  spin_times[tid] += __rdtsc() - t; \
}
#endif // TRACE_SPIN_TIME
#else
#define SPMP_LEVEL_SCHEDULE_WAIT SPMP_LEVEL_SCHEDULE_WAIT_DEFAULT
#endif // MEASURE_SPIN_TIME

#define SPMP_LEVEL_SCHEDULE_NOTIFY_DEFAULT { \
  taskFinished[task] = 1; \
}

#ifdef MEASURE_TASK_TIME
#define SPMP_LEVEL_SCHEDULE_NOTIFY { \
  task_time_sum[A.rowptr[i + 1] - A.rowptr[i]] += __rdtsc() - taskBegin; \
  task_time_cnt[A.rowptr[i + 1] - A.rowptr[i]]++; \
  SPMP_LEVEL_SCHEDULE_NOTIFY_DEFAULT; \
}
#else
#define SPMP_LEVEL_SCHEDULE_NOTIFY SPMP_LEVEL_SCHEDULE_NOTIFY_DEFAULT
#endif

class FusedGSAndSpMVSchedule;

/**
 * Estimate the time of each row for load balanced partitioning
 */
class CostFunction
{
public :
  virtual int getCostOf(int row) const { return 1; }
};

/**
 * cost[i] = prefixSum[i + 1] - prefixSum[i]
 *
 * Useful if cost is proportional to nnz of each row
 * (use prefixSum = rowptr)
 */
class PrefixSumCostFunction : public CostFunction
{
public :
  PrefixSumCostFunction(const int *prefixSum) : prefixSum(prefixSum) { };

  int getCostOf(int row) const { return prefixSum[row + 1] - prefixSum[row]; }

private :
  const int *prefixSum;
};

class LevelSchedule
{
public :
  LevelSchedule();
  virtual ~LevelSchedule();

  short *nparentsForward, *nparentsBackward;
  int ntasks;

  int **parentsForward, **parentsBackward;
  volatile int *taskFinished;

  std::vector<int> levIndices;
  int *origToThreadContPerm, *threadContToOrigPerm;
  std::vector<int> taskBoundaries; // begin and end rows
  std::vector<int> threadBoundaries; // begin and end task of each thread

  int *parentsBuf[2];

  // options
  bool useBarrier; // default: false
  bool transitiveReduction; // default: true
  bool fuseSpMV; // default: false
  bool aggregateForVectorization; // default: false in Xeon, true in Xeon Phi

  bool useMemoryPool;

  FusedGSAndSpMVSchedule *fusedSchedule;

  /**
   * Find levels and partition each level.
   * Remove intra-thread, duplicated, and transitive edges
   * Load balancing based on the num of non-zeros
   * using PrefixSumCostFunction(rowptr) cost function.
   *
   * @params A the matrix
   *
   * Assumptions:
   *  1. colidx of each row is partially sorted that
   *  lower triangular elements appear before diagptr and
   *  upper triangular elements appear after diagptr.
   *  2. symmetric non-zero pattern.
   *  Need this to have the same dependency graph for fwd/bwd solvers
   *  Need this to efficiently access outgoing edges w/o transpose
   *  If the input matrix is not structurally symmetric, compute
   *  A+A^T and pass it to this function.
   */
  void constructTaskGraph(CSR& A);

  void constructTaskGraph(CSR& A, const CostFunction& costFunction);

  /**
   * constructTaskGraph version that is not dependent on CSR type
   */
  void constructTaskGraph(
    int m, const int *rowptr, const int *colidx,
    const CostFunction& costFunction);

  /**
   * @params diagptr points to the index where the diag elem of each row locates
   *
   * @note in each row, lower diagonal values should appear before diag and
   *       upper diagonal values should appear after diag.
   */
  void constructTaskGraph(
    int m,
    const int *rowptr, const int *diagptr, const int *colidx,
    const CostFunction& costFuction);

  /**
   * constructTaskGraph version that supports a matrix distributed over multiple MPI nodes
   */
  void constructTaskGraph(
    int m,
    const int *rowptr, const int *diagptr, const int *extptr, const int *colidx,
    const CostFunction& costFunction);

  template<class T> T *allocate(size_t n)
  {
    if (useMemoryPool) {
      return MemoryPool::getSingleton()->allocate<T>(n);
    }
    else {
      return MALLOC(T, n);
    }
  }

  template<class T> T *allocateFront(size_t n)
  {
    if (useMemoryPool) {
      return MemoryPool::getSingleton()->allocateFront<T>(n);
    }
    else {
      return MALLOC(T, n);
    }
  }

protected:

  void init_();
};

/**
 * Schedule data structure when we schedule
 * GS and SpMV together
 */
class FusedGSAndSpMVSchedule
{
public :
  FusedGSAndSpMVSchedule(LevelSchedule *schedule);
  ~FusedGSAndSpMVSchedule();

  short *nparents; // number of parent GS tasks of each SpMV task
  int **parents;
  int *parentsBuf;

  LevelSchedule *gsSchedule_;
};

} // namespace SpMP
