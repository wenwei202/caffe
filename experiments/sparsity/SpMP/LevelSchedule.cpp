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

#include <cstring>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <omp.h>

#include "LevelSchedule.hpp"
#include "Utils.hpp"
#include "MemoryPool.hpp"

#ifdef LOADIMBA
#include "synk/loadimba.hpp"
#else
#include "synk/barrier.hpp"
#endif

using namespace std;

namespace SpMP
{

using namespace std;

#ifdef TRACE_TASK_TIMES
unsigned long long *taskTimes[NUM_MAX_THREADS] = { NULL };
unsigned long long taskTimeCnt[NUM_MAX_THREADS];
#endif

#ifdef MEASURE_SPIN_TIME
unsigned long long spin_times[NUM_MAX_THREADS] = {0};
#ifdef MEASURE_TASK_TIME
unsigned long long task_time_sum[65536] = {0};
int task_time_cnt[65536] = {0};
#endif

#ifdef TRACE_SPIN_TIME
unsigned long long *spin_traces[NUM_MAX_THREADS] = {NULL};
int spin_trace_counts[NUM_MAX_THREADS];
#endif
#endif

#ifdef TRACE_SPIN_TIME
void dumpTrace(const char *fileName)
{
  ofstream o(fileName);
  for (int ti = 0; ti < omp_get_max_threads(); ++ti) {
    for (int i = 1; i < spin_trace_counts[ti]; i += 2) {
      o <<
      "=Worker " << ti << "| type 0| start:" <<
      (spin_traces[ti][i] - spin_traces[ti][0]) << "| end:" <<
      (spin_traces[ti][i + 1] - spin_traces[ti][0]) << "| task 0" <<
      endl;
    }
  }
}
#endif

LevelSchedule::LevelSchedule()
 : nparentsForward(NULL), nparentsBackward(NULL),
 parentsForward(NULL), parentsBackward(NULL),
 taskFinished(NULL),
 useBarrier(false), transitiveReduction(true), fuseSpMV(false),
#ifdef __MIC__
 aggregateForVectorization(true),
#else
 aggregateForVectorization(false),
#endif
 useMemoryPool(false)
{
  init_();
}

LevelSchedule::~LevelSchedule() {
  if (!useMemoryPool) {
    FREE(nparentsForward);
    FREE(nparentsBackward);
    FREE(parentsForward);
    FREE(parentsBackward);

    int *tempTaskFinished = (int *)taskFinished;
    FREE(tempTaskFinished);

    FREE(origToThreadContPerm);
    FREE(threadContToOrigPerm);
    FREE(parentsBuf[0]);
    FREE(parentsBuf[1]);

    if (fusedSchedule) delete fusedSchedule;
  }
}

void LevelSchedule::init_()
{
  origToThreadContPerm= NULL;
  threadContToOrigPerm= NULL;
  parentsBuf[0] = parentsBuf[1] = NULL;
  fusedSchedule = NULL;
#ifdef LEVEL_CONT_LAYOUT
  levContToOrigPerm= NULL;
  origToLevContPerm= NULL;
  threadContToLevContPerm = NULL;
#endif
}

void LevelSchedule::constructTaskGraph(CSR& A)
{
  constructTaskGraph(A, PrefixSumCostFunction(A.rowptr));
}

template<int BASE = 0>
static int *constructDiagPtr_(
  int m, const int *rowptr, const int *colidx)
{
  int *diagptr = MALLOC(int, m);
#pragma omp parallel for
  for (int i = 0; i < m; ++i) {
    for (int j = rowptr[i] - BASE; j < rowptr[i + 1] - BASE; ++j) {
      if (colidx[j] - BASE >= i) {
        diagptr[i] = j + BASE;
        break;
      }
    }
  }
  return diagptr;
}

int *constructDiagPtr_(int m, const int *rowptr, const int *colidx, int base)
{
  if (0 == base) {
    return constructDiagPtr_<0>(m, rowptr, colidx);
  }
  else {
    assert(1 == base);
    return constructDiagPtr_<1>(m, rowptr, colidx);
  }
}

void CSR::constructDiagPtr()
{
  if (!diagptr) {
    diagptr = constructDiagPtr_(m, rowptr, colidx, getBase());
  }
}

void LevelSchedule::constructTaskGraph(CSR& A, const CostFunction& costFunction)
{
  A.constructDiagPtr();
  assert(A.isSymmetric(false));
  constructTaskGraph(A.m, A.rowptr, A.diagptr, A.extptr, A.colidx, costFunction);
}

void LevelSchedule::constructTaskGraph(
  int m, const int *rowptr, const int *colidx, const CostFunction& costFunction)
{
  int *diagptr = constructDiagPtr_(m, rowptr, colidx, rowptr[0]);

  constructTaskGraph(m, rowptr, diagptr, NULL, colidx, costFunction);

  FREE(diagptr);
}

void LevelSchedule::constructTaskGraph(
  int m, const int *rowptr, const int *diagptr, const int *colidx,
  const CostFunction& costFunction)
{
  constructTaskGraph(m, rowptr, diagptr, NULL, colidx, costFunction);
}

#define NUM_MAX_THREADS (2048)

template<int BASE = 0>
static
void findLevels_(
  int *qTail, int *nnzPrefixSum, int *nparents, int **q, int **rowPtrs,
  const int *diagptr, const int *extptr,
  const int *colidx,
  int *reversePerm, vector<int>& levIndices,
  int m,
  unsigned long long &tt2, unsigned long long &tt3, unsigned long long &tt4, unsigned long long &tt5, unsigned long long &tt6)
{
  colidx -= BASE;

  int qTailPrefixSum[NUM_MAX_THREADS] = { 0 };
  volatile int endSynchronized[1] = { 0 };

#ifdef LOADIMBA
  synk::LoadImba *bar = synk::LoadImba::getInstance();
#else
  synk::Barrier *bar = synk::Barrier::getInstance();
#endif

#pragma omp parallel
  {
  int tid = omp_get_thread_num();
  int nthreads = omp_get_num_threads();

  bool prevIterSerial = false;
  unsigned long long tTemp = __rdtsc();

  int end = 0;

  while (true) { // until all nodes are visited
    bool currIterSerial = false;
    int begin = end;

    if (prevIterSerial) {
      if (0 == tid) {
        if (qTail[0] == 0) {
          bar->wait(tid); // wake up other threads
          break;
        }
#define THRESH 64
        currIterSerial = qTail[0] < THRESH*2;
        if (!currIterSerial) {
          *endSynchronized = begin;
          bar->wait(tid); // wake up other threads
          end = begin + qTail[0];

          qTailPrefixSum[1] = qTail[0];
          bar->wait(tid);
        }
        else {
          end = begin + qTail[0];
        }
      }
      else {
        currIterSerial = false;

        begin = *endSynchronized;
        end = begin + qTail[0];
        qTailPrefixSum[tid + 1] = qTail[0];
        nnzPrefixSum[tid + 1] = nnzPrefixSum[1];
        bar->wait(tid);
      }
    }
    else {
      tTemp = __rdtsc();
      bar->wait(tid);
      if (0 == tid) tt2 += __rdtsc() - tTemp;

      tTemp = __rdtsc();

      // compute prefix sum of # of discovered nodes and nnzs associated with them
#pragma omp single
      {
        for (int t = 0; t < nthreads; ++t) {
          qTailPrefixSum[t + 1] = qTailPrefixSum[t] + qTail[t*16];
          nnzPrefixSum[t + 1] += nnzPrefixSum[t];
        }
      }

      if (0 == tid) tt3 += __rdtsc() - tTemp;

      if (qTailPrefixSum[nthreads] == 0) break;
      currIterSerial = qTailPrefixSum[nthreads] < THRESH;

      tTemp = __rdtsc();

      int begin = end;
      end = begin + qTailPrefixSum[nthreads];

      int iPerThread = (qTailPrefixSum[nthreads] + nthreads - 1)/nthreads;
      int iBegin = min(begin + iPerThread*tid, end);
      int iEnd = min(iBegin + iPerThread, end);

      int tBegin = upper_bound(
          qTailPrefixSum, qTailPrefixSum + nthreads + 1,
          iPerThread*tid) -
        qTailPrefixSum - 1;
      int tEnd = upper_bound(
          qTailPrefixSum, qTailPrefixSum + nthreads + 1,
          iPerThread*(tid + 1)) -
        qTailPrefixSum - 1;

      for (int t = tBegin; t <= min(tEnd, nthreads - 1); ++t) {
        int b = max(0, iBegin - begin - qTailPrefixSum[t]);
        int e = min((int)qTail[t*16], iEnd - begin - qTailPrefixSum[t]);

        memcpy(
          &reversePerm[0] + begin + qTailPrefixSum[t] + b,
          q[t] + b,
          (e - b)*sizeof(int));
      }
    }

    if (0 == tid) {
      levIndices.push_back(end);
    }

    if (currIterSerial) {
      if (!prevIterSerial) {
        bar->wait(tid);
      }

      if (0 == tid) {
        tTemp = __rdtsc();

        int *tailPtr = &reversePerm[end];
        int *rowPtr = rowPtrs[tid];
        *rowPtr = 0;

        for (int i = begin; i < end; ++i) {
          int pred = reversePerm[i];
          int jBegin = diagptr[pred] + 1;
          int jEnd = extptr[pred];
          for (int j = jBegin; j < jEnd; ++j) {
            int succ = colidx[j] - BASE;
            if (succ >= m) continue; // for a "fat" matrices with halo
            --nparents[succ];
            assert(nparents[succ] >= 0);
            if (nparents[succ] == 0) {
              *tailPtr = succ;
              *(rowPtr + 1) = *rowPtr + extptr[succ] - diagptr[succ] - 1;

              ++tailPtr;
              ++rowPtr;
            }
          }
        } // for each ready row

        qTail[0] = tailPtr - &reversePerm[end];
        nnzPrefixSum[1] = *rowPtr;

        tt6 += __rdtsc() - tTemp;
      }
      else {
        assert(!prevIterSerial);
        bar->wait(tid); // waiting for the master thread
        if (qTail[0] == 0) break;
      }
    }
    else {
      int nnzPerThread = (nnzPrefixSum[nthreads] + nthreads - 1)/nthreads;
      int tBegin = upper_bound(
          nnzPrefixSum, nnzPrefixSum + nthreads + 1,
          nnzPerThread*tid) -
        nnzPrefixSum - 1;
      int tEnd = upper_bound(
          nnzPrefixSum, nnzPrefixSum + nthreads + 1,
          nnzPerThread*(tid + 1)) -
        nnzPrefixSum - 1;

      int iBegin, iEnd;
      if (0 == tid) {
        iBegin = 0;
      }
      else if (tBegin == nthreads) {
        iBegin = qTailPrefixSum[nthreads];
      }
      else {
        iBegin = upper_bound(
            rowPtrs[tBegin], rowPtrs[tBegin] + qTail[tBegin*16],
            nnzPerThread*tid - nnzPrefixSum[tBegin]) -
          rowPtrs[tBegin] - 1 +
          qTailPrefixSum[tBegin];
      }

      if (tEnd == nthreads) {
        iEnd = qTailPrefixSum[nthreads];
      }
      else {
        iEnd = upper_bound(
            rowPtrs[tEnd], rowPtrs[tEnd] + qTail[tEnd*16],
            nnzPerThread*(tid + 1) - nnzPrefixSum[tEnd]) -
          rowPtrs[tEnd] - 1 +
          qTailPrefixSum[tEnd];
      }

      iBegin += begin;
      iEnd += begin;

      if (0 == tid) tt4 += __rdtsc() - tTemp;

      tTemp = __rdtsc();
      bar->wait(tid);
      if (0 == tid) tt5 += __rdtsc() - tTemp;

      tTemp = __rdtsc();
      int *tailPtr = q[tid];
      int *rowPtr = rowPtrs[tid];
      *rowPtr = 0;

      for (int i = iBegin; i < iEnd; ++i) {
        int pred = reversePerm[i];
        int jBegin = diagptr[pred] + 1;
        int jEnd = extptr[pred];
        for (int j = jBegin; j < jEnd; ++j) {
          int succ = colidx[j] - BASE;
          if (succ >= m) continue; // for a "fat" matrices with halo
          if (__sync_fetch_and_add(nparents + succ, -1) == 1) {
            *tailPtr = succ;
            *(rowPtr + 1) =
              *rowPtr + extptr[succ] - diagptr[succ] - 1;

            ++tailPtr;
            ++rowPtr;
          }
          assert(nparents[succ] >= 0);
        }
      } // for each ready row

      qTail[tid*16] = tailPtr - q[tid];
      nnzPrefixSum[tid + 1] = *rowPtr;
      if (0 == tid) tt6 += __rdtsc() - tTemp;
    }

    prevIterSerial = currIterSerial;
  } // while true
  } // omp parallel

#ifndef NDEBUG
  for (int i = 0; i < m; ++i) {
    assert(0 == nparents[i]);
  }
#endif
  assert(isPerm(reversePerm, m));
}

// This routine doesn't depend on that colidx of each row is fully sorted.
// It only needs to be partially sorted that those smaller than its row index
// appear before diagptr, those larger than # of rows appear after extptr,
// and all others appear in between.
template<int BASE = 0>
static
void findLevels_(
  LevelSchedule *schedule,
  int m, const int *rowptr, const int *diagptr, const int *extptr, const int *colidx,
  const CostFunction& costFunction)
{
  vector<int>& levIndices = schedule->levIndices;
  vector<int>& taskBoundaries = schedule->taskBoundaries;
  vector<int>& threadBoundaries = schedule->threadBoundaries;

  bool useBarrier = schedule->useBarrier;
  bool aggregateForVectorization = schedule->aggregateForVectorization;

  bool useMemoryPool = schedule->useMemoryPool;

#ifndef NDEBUG
  CSR A(m, m, (int *)rowptr, (int *)colidx, (double *)NULL);
  A.extptr = (int *)extptr;
  assert(A.isSymmetric(false, true));
  A.extptr = NULL;
  A.~CSR();
#endif

  // extptr points to the beginning of non-local columns (when MPI is used).
  // When MPI is not used, exptr will be set to NULL and we use rowptr + 1 as
  // extptr.
  if (NULL == extptr) {
    extptr = rowptr + 1;
  }

  // check columns are partially sorted that lower triangular parts appear before
  // diagptr and upper triangular parts appear after diagptr
#ifndef NDEBUG
  for (int i = 0; i < m; ++i) {
    for (int j = rowptr[i] - BASE; j < diagptr[i] - BASE; ++j) {
      assert(colidx[j] - BASE < i);
    }
    for (int j = diagptr[i] - BASE; j < extptr[i] - BASE; ++j) {
      assert(colidx[j] - BASE >= i);
    }
    for (int j = extptr[i] - BASE; j < rowptr[i] - BASE; ++j) {
      assert(colidx[j] - BASE >= m);
    }
  }
#endif

  unsigned long long tt0 = 0, tt1 = 0, tt2 = 0, tt3 = 0, tt4 = 0, tt5 = 0, tt6 = 0;

  unsigned long long t = __rdtsc();
  int n = m;
  int nthreads = omp_get_max_threads();

  MemoryPool *memoryPool = MemoryPool::getSingleton();

  int nAligned = (n + 15)/16*16;
#ifndef LEVEL_CONT_LAYOUT
  int *levContToOrigPerm;
#endif
  levContToOrigPerm = schedule->allocate<int>(nAligned + 128);

  size_t bufferBegin = memoryPool->getTail();
  int *buffer = schedule->allocate<int>(5*max(nthreads, n + nthreads - 1) + NUM_MAX_THREADS*(2 + 16 + 2 + 2)); // max required in case n == 0
  assert(buffer);
  int *tempBuffer = buffer + max(nthreads, n + nthreads - 1)*4;
    // between buffer and tempBuffer, the followings are allocated
    // q : max(nthreads, n + nthreads - 1)*2
    // rowPtrs : max(nthreads, n + nthreads - 1)*2

  int *nparents = tempBuffer; tempBuffer += n;
  int **q = (int **)tempBuffer; tempBuffer += NUM_MAX_THREADS*(sizeof(int *)/sizeof(int));
  int *qTail = tempBuffer; tempBuffer += NUM_MAX_THREADS*16;
  int **rowPtrs = (int **)tempBuffer; tempBuffer += NUM_MAX_THREADS*(sizeof(int *)/sizeof(int));
  int *nnzPrefixSum = tempBuffer; tempBuffer += NUM_MAX_THREADS*2;

  nnzPrefixSum[0] = 0;
  nnzPrefixSum[nthreads + 1] = 0;

  levIndices.clear();
  levIndices.push_back(0);

  tt0 = __rdtsc() - t;
  unsigned long long ttt;

#pragma omp parallel num_threads(nthreads)
  {
  int tid = omp_get_thread_num();
  if (0 == tid) ttt = __rdtsc();
  unsigned long long tTemp = __rdtsc();

  q[tid] = buffer + max(nthreads, n + nthreads - 1)/nthreads*2*tid;
  rowPtrs[tid] = buffer + max(nthreads, n + nthreads - 1)*2 + max(nthreads, n + nthreads - 1)/nthreads*2*tid;

  int *tailPtr = q[tid];
  int *rowPtr = rowPtrs[tid];
  *rowPtr = 0;

#pragma omp for
  for (int i = 0; i < n; ++i) {
    nparents[i] = diagptr[i] - rowptr[i];
    if (nparents[i] == 0) {
      *tailPtr = i;
      *(rowPtr + 1) = *rowPtr + extptr[i] - diagptr[i] - 1;

      ++tailPtr;
      ++rowPtr;
    }
  }

  qTail[tid*16] = tailPtr - q[tid];
  nnzPrefixSum[tid + 1] = *rowPtr;

  if (0 == tid) tt1 = __rdtsc() - tTemp;
  } // omp parallel

  assert(q[0]);
  findLevels_<BASE>(
    qTail, nnzPrefixSum, nparents, q, rowPtrs,
    diagptr, extptr, colidx,
    levContToOrigPerm, levIndices, m,
    tt2, tt3, tt4, tt5, tt6);

  ttt = __rdtsc() - ttt;

//#define PRINT_TIME_BREAKDOWN
#ifdef PRINT_TIME_BREAKDOWN
  printf("t1 = %f\n", (__rdtsc() - t)/get_cpu_freq());
  printf("ttt = %f\n", ttt/get_cpu_freq());
  printf("tt0 = %f (init)\n", tt0/get_cpu_freq());
  printf("tt1 = %f (first level)\n", tt1/get_cpu_freq());
  printf("tt2 = %f (first barrier)\n", tt2/get_cpu_freq());
  printf("tt3 = %f (prefix sum)\n", tt3/get_cpu_freq());
  printf("tt4 = %f (copy)\n", tt4/get_cpu_freq());
  printf("tt5 = %f (second barrier)\n", tt5/get_cpu_freq());
  printf("tt6 = %f (traverse)\n", tt6/get_cpu_freq());
#undef PRINT_TIME_BREAKDOWN
#endif
  t = __rdtsc();

  if (useMemoryPool) {
    memoryPool->setTail(bufferBegin);
    buffer = NULL;
  }
  else {
    FREE(buffer);
  }

  //int ippNumThreads = 0;
  //ippGetNumThreads(&ippNumThreads);
  //printf("ippGetNumthreads = %d\n", ippNumThreads);
  //ippSetNumThreads(1);

  int *reversePerm = levContToOrigPerm;

  size_t taskRowsEnd = memoryPool->getHead();
  pair<int, int> *taskRows =
    schedule->allocateFront<pair<int, int> >(nthreads*levIndices.size() - 1);

  /*copy(levIndices.begin(), levIndices.end(), ostream_iterator<int>(cout, " "));
  printf("\n");*/

  size_t bBufferBegin = memoryPool->getTail();
  int *bBuffer = schedule->allocate<int>(n + nthreads);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int l = 0; l < levIndices.size() - 1; ++l) {
    int levelBegin = levIndices[l], levelEnd = levIndices[l + 1];
    int *b = &bBuffer[0];

    sort(&reversePerm[levelBegin], &reversePerm[levelEnd]);

    // partition this level.
    int nnz = 0;
    for (int i = levelBegin; i < levelEnd; ++i) {
      b[i] = nnz;
      nnz += costFunction.getCostOf(reversePerm[i]);
    }
    int nnzPerThread = (nnz + nthreads - 1)/nthreads;

    int prevEnd = levelBegin;
    int r = levelBegin;
    nnz = 0;
    int t;
    for (t = 0; t < nthreads; ++t) {
      int newr = lower_bound(&b[r], &b[levelEnd], (t + 1)*nnzPerThread) - &b[0];
      if (aggregateForVectorization) {
        // make task size a multiple of 8 as much as possible
        if (0 == t) {
          r = min(r + (newr - r + 7)/8*8, levelEnd);
        }
        else {
          r = min(r - 1 + (newr - r + 1 + 7)/8*8, levelEnd);
        }
      }
      else {
        r = newr;
      }
      //for ( ; r < levelEnd && nnz < (t + 1)*nnzPerThread; r++) {
        //nnz += L.rowptr[reversePerm[r] + 1] - L.rowptr[reversePerm[r]];
      //}
      //assert(r == r2);

      int begin = prevEnd;
      int end = min(r, levelEnd);
      prevEnd = end;

      taskRows[t*(levIndices.size() - 1) + l] = make_pair(begin, end);
      if (aggregateForVectorization && end >= levelEnd) break;

      //printf("(%d, %d) (%d, %d) (%d, %d)\n", l, t, begin, end, levelBegin, levelEnd);
      //if (r < levelEnd)
        //nnz += L.rowptr[reversePerm[r] + 1] - L.rowptr[reversePerm[r]];
      ++r; // make sure first threads execute some tasks -> improve locality
    } // for each thread

    if (aggregateForVectorization) {
      // shift tasks to the last threads so that earlier threads
      // have tasks whose number of rows is a multiple of 8.

      // we have (t + 1) non-empty tasks in this level
      // copy [0, t] to [nthreads - 1 - t, nthreads - 1]
      for (int i = t; i >= 0; --i) {
        taskRows[(nthreads - 1 - t + i)*(levIndices.size() - 1) + l] =
          taskRows[i*(levIndices.size() - 1) + l];
      }
      // fill [0, nthreads - 1 - t) empty
      for (int i = 0; i < nthreads - t - 1; ++i) {
        taskRows[i*(levIndices.size() - 1) + l] = make_pair(levelBegin, levelBegin);
      }
    }
  } // for each level

  if (useMemoryPool) {
    memoryPool->setTail(bBufferBegin);
    bBuffer = NULL;
  }
  else {
    FREE(bBuffer);
  }

  //printf("t2 = %f\n", (__rdtsc() - t)/get_cpu_freq());

  // new permutation so that rows for a thread are contiguous
  int i = 0;
  vector<int> rowPartialSum(nthreads + 1);
  rowPartialSum[0] = 0;

  threadBoundaries.resize(nthreads + 1);
  threadBoundaries[0] = 0;
  if (useBarrier) {
#pragma omp parallel for
    for (int tid = 0; tid < nthreads; ++tid) {
      int sum = 0;
      int cnt = 0;
      for (int i = 0; i < levIndices.size() - 1; ++i) {
        int diff = taskRows[tid*(levIndices.size() - 1) + i].second - taskRows[tid*(levIndices.size() - 1) + i].first;
        sum += diff;
        ++cnt;
      }
      rowPartialSum[tid + 1] = sum;
      threadBoundaries[tid + 1] = cnt;
    }
  }
  else {
#pragma omp parallel for
    for (int tid = 0; tid < nthreads; ++tid) {
      int sum = 0;
      int cnt = 0;
      for (int i = 0; i < levIndices.size() - 1; ++i) {
        int diff = taskRows[tid*(levIndices.size() - 1) + i].second - taskRows[tid*(levIndices.size() - 1) + i].first;
        sum += diff;
        if (diff) ++cnt;
      }
      rowPartialSum[tid + 1] = sum;
      threadBoundaries[tid + 1] = cnt;
    }
  }

  for (int tid = 0; tid < nthreads; ++tid) {
    rowPartialSum[tid + 1] += rowPartialSum[tid];
    threadBoundaries[tid + 1] += threadBoundaries[tid];
  }

  FREE(schedule->origToThreadContPerm);
  FREE(schedule->threadContToOrigPerm);
  schedule->origToThreadContPerm = schedule->allocate<int>(m + 64);
  schedule->threadContToOrigPerm = schedule->allocate<int>(m + 64);
  int *origToThreadContPerm = schedule->origToThreadContPerm;
  int *threadContToOrigPerm = schedule->threadContToOrigPerm;

  taskBoundaries.resize(threadBoundaries[nthreads] + 1);

  if (useBarrier) {
#pragma omp parallel for
    for (int tid = 0; tid < nthreads; ++tid) {
      int rowOffset = rowPartialSum[tid];
      int taskOffset = threadBoundaries[tid];
      for (int i = 0; i < levIndices.size() - 1; ++i) {
        taskBoundaries[taskOffset] = rowOffset;
        ++taskOffset;

        for (int j = taskRows[tid*(levIndices.size() - 1) + i].first; j < taskRows[tid*(levIndices.size() - 1) + i].second; ++j) {
          origToThreadContPerm[reversePerm[j]] = rowOffset;
          threadContToOrigPerm[rowOffset] = reversePerm[j];
          rowOffset++;
        }
      } // for each task
    } // for each thread
  } // useBarrier
  else {
#pragma omp parallel for
    for (int tid = 0; tid < nthreads; ++tid) {
      int rowOffset = rowPartialSum[tid];
      int taskOffset = threadBoundaries[tid];
      for (int i = 0; i < levIndices.size() - 1; ++i) {
        if (taskRows[tid*(levIndices.size() - 1) + i].second > taskRows[tid*(levIndices.size() - 1) + i].first) {
          taskBoundaries[taskOffset] = rowOffset;
          ++taskOffset;

          for (int j = taskRows[tid*(levIndices.size() - 1) + i].first; j < taskRows[tid*(levIndices.size() - 1) + i].second; ++j) {
            origToThreadContPerm[reversePerm[j]] = rowOffset;
            assert(reversePerm[j] >= 0);
            threadContToOrigPerm[rowOffset] = reversePerm[j];
            rowOffset++;
          }
        }
      } // for each task
    } // for each thread
  } // !useBarrier
  taskBoundaries[threadBoundaries[nthreads]] = m;

  assert(isPerm(origToThreadContPerm, m));
  assert(isPerm(threadContToOrigPerm, m));

  if (!useMemoryPool) {
    FREE(taskRows);
  }
  else {
    memoryPool->setHead(taskRowsEnd);
    taskRows = NULL;
  }

#ifdef LEVEL_CONT_LAYOUT
  origToLevContPerm= MALLOC(int, n);
  threadContToLevContPerm = MALLOC(int, n);
  getInversePerm(origToLevContPerm, levContToOrigPerm, n);

#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    threadContToLevContPerm[i] =
      origToLevContPerm[threadContToOrigPerm[i]];
  }
#else
  if (!useMemoryPool) {
    FREE(levContToOrigPerm);
  }
#endif

  schedule->ntasks = threadBoundaries.back();
}

template<int BASE = 0>
static
void constructTaskGraph_(
  LevelSchedule *schedule,
  int m,
  const int *rowptr, const int *diagptr, const int *extptr, const int *colidx,
  const CostFunction& costFunction)
{
  vector<int>& levIndices = schedule->levIndices;
  vector<int>& taskBoundaries = schedule->taskBoundaries;
  vector<int>& threadBoundaries = schedule->threadBoundaries;

  int **parentsBuf = schedule->parentsBuf;

  bool transitiveReduction = schedule->transitiveReduction;
  bool fuseSpMV = schedule->fuseSpMV;

  bool useMemoryPool = schedule->useMemoryPool;

  findLevels_<BASE>(schedule, m, rowptr, diagptr, extptr, colidx, costFunction);

  int ntasks = schedule->ntasks;
  int *threadContToOrigPerm = schedule->threadContToOrigPerm;

  if (NULL == extptr) {
    extptr = rowptr + 1;
  }
  colidx -= BASE;

  int nnz = rowptr[m] - BASE;

  unsigned long long t;
#ifdef _OPENMP
  int nthreads = omp_get_max_threads();
#else
  int nthreads = 1;
#endif
  double spmvTime = nnz*12/60/1e9;

  MemoryPool *memoryPool = MemoryPool::getSingleton();
  size_t origRowIdToTaskIdEnd = memoryPool->getHead();
  int *origRowIdToTaskId = schedule->allocateFront<int>(m);
#pragma omp parallel for num_threads(nthreads)
  for (int task = 0; task < ntasks; ++task) {
    for (int row = taskBoundaries[task]; row < taskBoundaries[task + 1]; ++row) {
      origRowIdToTaskId[threadContToOrigPerm[row]] = task;
    }
  }

  int isBackwardBegin = 0;
  int isBackwardEnd = 1;

  schedule->nparentsForward = schedule->allocate<short>(ntasks);
  schedule->parentsForward = schedule->allocate<int *>(ntasks);
  short *nparentsForward = schedule->nparentsForward;
  int **parentsForward = schedule->parentsForward;

  schedule->nparentsBackward = schedule->allocate<short>(ntasks);
  schedule->parentsBackward = schedule->allocate<int *>(ntasks);
  short *nparentsBackward = schedule->nparentsBackward;
  int **parentsBackward = schedule->parentsBackward;

  schedule->taskFinished = schedule->allocate<int>(ntasks);
  volatile int *taskFinished = schedule->taskFinished;

#pragma omp parallel for
  for (int i = 0; i < ntasks; ++i) {
    nparentsForward[i] = 0;
    parentsForward[i] = NULL;

    nparentsBackward[i] = 0;
    parentsBackward[i] = NULL;

    taskFinished[i] = 0;
  }

#ifdef LOG_MEMORY_ALLOCATION
  printf("Allocate %d MB in %s\n", (sizeof(short)*ntasks + sizeof(int *)*ntasks + sizeof(int)*ntasks)/1024/1024, __func__);
  fflush(stdout);
#endif

  size_t childrenBufEnd = 0;
  short *nparentsTemp;
  vector<vector<int> > tempParents;

  unsigned long long barrierTimes[NUM_MAX_THREADS] = { 0 };

  t = __rdtsc();

  size_t adjacencyBegin = memoryPool->getTail();
  CSR taskAdjacency, taskInvAdjacency;
  taskAdjacency.rowptr = schedule->allocate<int>(ntasks + 1);
  taskAdjacency.colidx = schedule->allocate<int>((nnz - m)/2 + m);
  taskInvAdjacency.rowptr = schedule->allocate<int>(ntasks + 1);
  taskInvAdjacency.colidx = schedule->allocate<int>((nnz - m)/2 + m + ntasks);

  int *taskInvAdjacencyLengths = schedule->allocate<int>(ntasks);
#pragma omp parallel for num_threads(nthreads)
  for (int i = 0; i < ntasks; ++i) {
    taskInvAdjacencyLengths[i] = 0;
  }

#ifdef LOG_MEMORY_ALLOCATION
  printf(
    "Allocate %d MB in %s\n", 
    sizeof(int)*(ntasks*3 + (nnz - m) + 2*m)/1024/1024,
    __func__);
  fflush(stdout);
#endif

  vector<int> nnzPerRow(ntasks);
  
  int perThreadOrigRowPtrSum[NUM_MAX_THREADS] = { 0 };
  int perThreadOrigInvRowPtrSum[NUM_MAX_THREADS] = { 0 };
  int perThreadRowPtrSum[NUM_MAX_THREADS] = { 0 };
  int numNewEdges[NUM_MAX_THREADS] = { 0 };
  int *childrenBuf;
  int **children;
  short *numOfChildren;

  int v1Boundaries[NUM_MAX_THREADS] = { 0 }, v1Boundaries2[NUM_MAX_THREADS] = { 0 };

#ifdef _OPENMP
#pragma omp parallel num_threads(nthreads)
#endif
  {
  int tid = omp_get_thread_num();
  int v1PerThread = (ntasks + nthreads - 1)/nthreads;
  int v1Begin = min(tid*v1PerThread, ntasks);
  int v1End = min(v1Begin + v1PerThread, ntasks);

  // count nnzs to load balance
  int origRowPtrCnt = 0, origInvRowPtrCnt = 0;
  for (int v1 = v1Begin; v1 < v1End; ++v1) {
    int size = 0;
    int invSize = 0;
    for (int i1Perm = taskBoundaries[v1]; i1Perm < taskBoundaries[v1 + 1]; ++i1Perm) {
      int i1 = threadContToOrigPerm[i1Perm];
      size += extptr[i1] - diagptr[i1] - 1;
      invSize += diagptr[i1] - rowptr[i1] + 1; // additional 1 for intra-thread
    }
    taskAdjacency.rowptr[v1] = origRowPtrCnt;
    taskInvAdjacency.rowptr[v1] = origInvRowPtrCnt;
    origRowPtrCnt += size;
    origInvRowPtrCnt += invSize;
    nnzPerRow[v1] = size;
  }
  perThreadOrigRowPtrSum[tid + 1] = origRowPtrCnt;
  perThreadOrigInvRowPtrSum[tid + 1] = origInvRowPtrCnt;

  // compute prefix sums
#ifdef _OPENMP
#pragma omp barrier
#pragma omp single
#endif
  {
    for (int tid = 0; tid < nthreads; ++tid) {
      perThreadOrigRowPtrSum[tid + 1] += perThreadOrigRowPtrSum[tid];
      perThreadOrigInvRowPtrSum[tid + 1] += perThreadOrigInvRowPtrSum[tid];
    }
  }

  for (int v1 = v1Begin; v1 < v1End; ++v1) {
    taskInvAdjacency.rowptr[v1] += perThreadOrigInvRowPtrSum[tid];
  }

  // load-balanced partition
  if (tid == nthreads - 1) {
    v1Boundaries[tid + 1] = ntasks;
  }
  else {
    int nnzPerThread = ((nnz - m)/2 + nthreads - 1)/nthreads;
    int boundaryTid = 
      upper_bound(
        perThreadOrigRowPtrSum, perThreadOrigRowPtrSum + nthreads - 1,
        nnzPerThread*(tid + 1)) -
      perThreadOrigRowPtrSum - 1;

    v1Boundaries[tid + 1] =
      upper_bound(
        taskAdjacency.rowptr + min(boundaryTid*v1PerThread, ntasks),
        taskAdjacency.rowptr + min((boundaryTid + 1)*v1PerThread, ntasks),
        nnzPerThread*(tid + 1) - perThreadOrigRowPtrSum[boundaryTid]) -
      taskAdjacency.rowptr - 1;
    v1Boundaries[tid + 1] = max(0, v1Boundaries[tid + 1]);
  }

#ifdef _OPENMP
#pragma omp barrier
#endif

  v1Begin = v1Boundaries[tid];
  v1End = v1Boundaries[tid + 1];

  int t1Begin = distance(
    threadBoundaries.begin(),
    upper_bound(threadBoundaries.begin(), threadBoundaries.end(), v1Begin)) - 1;
  int t1End = distance(
    threadBoundaries.begin(),
    upper_bound(threadBoundaries.begin(), threadBoundaries.end(), v1End)) - 1;
  t1End = min(t1End, nthreads - 1);

  origRowPtrCnt = 0;
  int maxSize = 0;
  for (int v1 = v1Begin; v1 < v1End; ++v1) {
    assert(v1 >= 0 && v1 < ntasks);
    int size = nnzPerRow[v1];
    maxSize = max(size, maxSize);
    origRowPtrCnt += size;
  }
  perThreadOrigRowPtrSum[tid + 1] = origRowPtrCnt;

#ifdef _OPENMP
#pragma omp barrier
#pragma omp single
#endif
  {
    for (int tid = 0; tid < nthreads; ++tid) {
      perThreadOrigRowPtrSum[tid + 1] += perThreadOrigRowPtrSum[tid];
    }
  }

  // construct task dependency graph
  unsigned long long t11 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0;
  int rowPtrCnt = 0;
  vector<int> i2s(maxSize);
  int maxL = 0;
  for (int t1 = t1Begin; t1 <= t1End; ++t1) {
    for (int v1 = max(threadBoundaries[t1], v1Begin); v1 < min(threadBoundaries[t1 + 1], v1End); ++v1) {
      unsigned long long tempTimer = __rdtsc();

      int
        boundaryBegin = threadBoundaries[t1],
        boundaryEnd = threadBoundaries[t1 + 1];

      int size = 0;
      for (int i1Perm = taskBoundaries[v1]; i1Perm < taskBoundaries[v1 + 1]; ++i1Perm) {
        int i1 = threadContToOrigPerm[i1Perm];

        int begin = diagptr[i1] + 1;
        int end = extptr[i1];
        for (int j = begin; j < end; ++j) {
          int v = origRowIdToTaskId[colidx[j] - BASE];
          if (v < boundaryBegin || v >= boundaryEnd) {
            i2s[size] = v;
            ++size;
          }
          //else {
            //__sync_fetch_and_add(&intraThreadDependencyCount, 1);
          //}
        }
      }

      t11 += __rdtsc() - tempTimer;
      tempTimer = __rdtsc();

      sort(i2s.begin(), i2s.begin() + size);

      t2 += __rdtsc() - tempTimer;
      tempTimer = __rdtsc();

      int oldV2 = -1;

      taskAdjacency.rowptr[v1] = perThreadOrigRowPtrSum[tid] + rowPtrCnt;
      int *adjList = taskAdjacency.colidx + perThreadOrigRowPtrSum[tid] + rowPtrCnt;
      int l = 0;
      bool intraAdded = v1 == threadBoundaries[t1 + 1] - 1;
      for (int i2 = 0; i2 < size; ++i2) {
        int v2 = i2s[i2];
        if (v2 == oldV2) continue;
        oldV2 = v2;

        adjList[l] = v2;
        ++l;

        if (!intraAdded && transitiveReduction && v2 >= v1 + 1) {
          if (v2 > v1 + 1) {
            int tempL = __sync_fetch_and_add(&taskInvAdjacencyLengths[v1 + 1], 1);
            taskInvAdjacency.colidx[taskInvAdjacency.rowptr[v1 + 1] + tempL] = v1;
          }
          intraAdded = true;
        }
        int tempL = __sync_fetch_and_add(&taskInvAdjacencyLengths[v2], 1);
        taskInvAdjacency.colidx[taskInvAdjacency.rowptr[v2] + tempL] = v1;
      }
      if (!intraAdded && transitiveReduction) {
        int tempL = __sync_fetch_and_add(&taskInvAdjacencyLengths[v1 + 1], 1);
        taskInvAdjacency.colidx[taskInvAdjacency.rowptr[v1 + 1] + tempL] = v1;
      }

      t3 += __rdtsc() - tempTimer;
      rowPtrCnt += l;
      nnzPerRow[v1] = l;
      maxL = max(maxL, l);
    }
  }

  perThreadRowPtrSum[tid + 1] = rowPtrCnt;

  unsigned long long tempTimer = __rdtsc();

#ifdef _OPENMP
#pragma omp barrier
#endif

  t4 = __rdtsc() - tempTimer;
  barrierTimes[tid] = t4;
  tempTimer = __rdtsc();

  // compute prefix sum
#ifdef _OPENMP
#pragma omp single
#endif
  {
    for (int tid = 0; tid < nthreads; ++tid) {
      perThreadRowPtrSum[tid + 1] += perThreadRowPtrSum[tid];
    }
    taskAdjacency.rowptr[ntasks] = perThreadOrigRowPtrSum[nthreads];
  }

  for (int task = v1Begin; task < v1End; ++task) {
    sort(
      taskInvAdjacency.colidx + taskInvAdjacency.rowptr[task],
      taskInvAdjacency.colidx + taskInvAdjacency.rowptr[task] + taskInvAdjacencyLengths[task]);
  }

  t5 = __rdtsc() - tempTimer;

#ifdef _OPENMP
  // construct FusedGSAndSpMVSchedule
  if (fuseSpMV) {
#pragma omp barrier
#pragma omp single // FIXME - parallelize this
    {
      schedule->fusedSchedule = new FusedGSAndSpMVSchedule(schedule);

      tempParents.resize(ntasks);
      for (int task = 0; task < ntasks; ++task) {
        for (int j = 0; j < nnzPerRow[task]; ++j) {
          int child = taskAdjacency.colidx[taskAdjacency.rowptr[task] + j];
          if (tempParents[child].empty()) {
            tempParents[child].push_back(task);
          }
          else {
            // if there's already a parent, append only if the new parent is assigned
            // to a different thread
            int t = distance(
              threadBoundaries.begin(),
              upper_bound(threadBoundaries.begin(), threadBoundaries.end(), task)) - 1;
            int tPrev = distance(
              threadBoundaries.begin(),
              upper_bound(threadBoundaries.begin(), threadBoundaries.end(), tempParents[child].back())) - 1;
            if (t != tPrev) {
              tempParents[child].push_back(task);
            }
          }
        } // for each child of this task
      } // for each task
    } // omp single

      // add intra-thread dependency
#pragma omp for
    for (int t = 0; t < nthreads; ++t) {
      for (int task = threadBoundaries[t]; task < threadBoundaries[t + 1]; ++task) {
        int minParentTask = task;
        for (int i = taskBoundaries[task]; i < taskBoundaries[task + 1]; ++i) {
          int row = threadContToOrigPerm[i];
          int begin = rowptr[row], end = diagptr[row];
          for (int j = begin; j < end; ++j) {
            int parentTask = origRowIdToTaskId[colidx[j] - BASE];
            if (parentTask >= threadBoundaries[t] && parentTask < threadBoundaries[t + 1]) {
              if (parentTask < minParentTask) {
                minParentTask = parentTask;
              }
            }
#if 0 // ndef NDEBUG
            else {
              bool found = false;
              for (int k = 0; k < fusedSchedule->nparents[task]; ++k) {
                if (fusedSchedule->parents[task][k] == parentTask) {
                  found = true;
                  break;
                }
              }
              if (!found) {
                printf("%d->%d is missing, row = %d, parentRow = %d\n", parentTask, task, row, colidx[j]);
              }
              assert(found);
            }
#endif
          } // for each parent
        } // for each row

        tempParents[task].push_back(minParentTask);
        //fusedSchedule->parents[task][fusedSchedule->nparents[task]] = minParentTask;
        //fusedSchedule->nparents[task]++;
      } // for each task
    } // for each thread

    // remove transitive dependencies
    // v->u is redundant when
    // case a : v ...-> q -> p -> u
    //        (v is executed before q in the same thread)
    // case b : v -> q ...-> p -> u
#pragma omp for
    for (int u = 0; u < ntasks; ++u) {
      sort(tempParents[u].begin(), tempParents[u].end());

      vector<int> parentsToRemove;
      for (auto v = tempParents[u].begin(); v != tempParents[u].end(); ++v) {
        int tOfV = distance(
          threadBoundaries.begin(),
          upper_bound(threadBoundaries.begin(), threadBoundaries.end(), *v)) - 1;
        bool redundant = false;

        for (auto p = tempParents[u].begin(); p != tempParents[u].end(); ++p) {
          // case a
          for (int j = 0; j < nnzPerRow[*p]; ++j) {
            int q = taskAdjacency.colidx[taskAdjacency.rowptr[*p] + j];
            if (q >= threadBoundaries[tOfV] && q <= *v) {
              redundant = true;
              break;
            }
          }

          // case b : this happens very rarely
          /*int tOfP = distance(
            threadBoundaries.begin(),
            upper_bound(threadBoundaries.begin(), threadBoundaries.end(), *p)) - 1;

          for (int j = 0; j < taskInvAdjacencyLengths[*v]; ++j) {
            int q = taskInvAdjacency.colidx[taskInvAdjacency.rowptr[*v] + j];
            if (q >= *p && q < threadBoundaries[tOfP]) {
              redundant = true;
              break;
            }
          }*/
        }

        if (redundant) {
          parentsToRemove.push_back(*v);
        }
      } // for each parent v of u

      sort(parentsToRemove.begin(), parentsToRemove.end());
      vector<int> remainder;
      set_difference(
        tempParents[u].begin(), tempParents[u].end(),
        parentsToRemove.begin(), parentsToRemove.end(),
        back_inserter(remainder));
      tempParents[u].swap(remainder);
    } // for each u

#pragma omp single
    {
      FusedGSAndSpMVSchedule *fusedSchedule = schedule->fusedSchedule;
      fusedSchedule->parents = MALLOC(int *, ntasks);
      fusedSchedule->parentsBuf = MALLOC(int, perThreadOrigRowPtrSum[nthreads] + 2*ntasks);

      fusedSchedule->nparents = MALLOC(short, ntasks);
      memset((void *)fusedSchedule->nparents, 0, sizeof(short)*ntasks);

      int cnt = 0;
      for (int task = 0; task < ntasks; ++task) {
        fusedSchedule->parents[task] = fusedSchedule->parentsBuf + cnt;
        fusedSchedule->nparents[task] = tempParents[task].size();
        copy(tempParents[task].begin(), tempParents[task].end(), fusedSchedule->parents[task]);
        cnt += taskInvAdjacencyLengths[task];
      }

      /*int cnt = 0;
      for (int task = 0; task < ntasks; ++task) {
        fusedSchedule->parents[task] = fusedSchedule->parentsBuf + cnt;
        cnt += taskInvAdjacencyLengths[task] + 1;
      }
      assert(cnt <= perThreadOrigRowPtrSum[nthreads] + ntasks);

      for (int task = 0; task < ntasks; ++task) {
        for (int j = 0; j < nnzPerRow[task]; ++j) {
          int child = taskAdjacency.colidx[taskAdjacency.rowptr[task] + j];
          if (fusedSchedule->nparents[child] > 0) {
            int t = distance(
              threadBoundaries.begin(),
              upper_bound(threadBoundaries.begin(), threadBoundaries.end(), task)) - 1;
            int tPrev = distance(
              threadBoundaries.begin(),
              upper_bound(threadBoundaries.begin(), threadBoundaries.end(), fusedSchedule->parents[child][fusedSchedule->nparents[child] - 1])) - 1;
            if (t != tPrev) {
              fusedSchedule->parents[child][fusedSchedule->nparents[child]] = task;
              fusedSchedule->nparents[child]++;
            }
          }
          else {
            fusedSchedule->parents[child][fusedSchedule->nparents[child]] = task;
            fusedSchedule->nparents[child]++;
          }
        }
      } // for each task

      // add intra-thread dependency
      for (int t = 0; t < nthreads; ++t) {
        for (int task = threadBoundaries[t]; task < threadBoundaries[t + 1]; ++task) {
          int minParentTask = task;
          for (int i = taskBoundaries[task]; i < taskBoundaries[task + 1]; ++i) {
            int row = threadContToOrigPerm[i];
            int begin = rowptr[row], end = diagptr[row];
            for (int j = begin; j < end; ++j) {
              int parentTask = origRowIdToTaskId[colidx[j]];
              if (parentTask >= threadBoundaries[t] && parentTask < threadBoundaries[t + 1]) {
                if (parentTask < minParentTask) {
                  minParentTask = parentTask;
                }
              }
#if 0 // ndef NDEBUG
              else {
                bool found = false;
                for (int k = 0; k < fusedSchedule->nparents[task]; ++k) {
                  if (fusedSchedule->parents[task][k] == parentTask) {
                    found = true;
                    break;
                  }
                }
                if (!found) {
                  printf("%d->%d is missing, row = %d, parentRow = %d\n", parentTask, task, row, colidx[j]);
                }
                assert(found);
              }
#endif
            } // for each parent
          } // for each row

          fusedSchedule->parents[task][fusedSchedule->nparents[task]] = minParentTask;
          fusedSchedule->nparents[task]++;
        } // for each task
      } // for each thread*/

//#define PRINT_AVG_DEGREE_OF_SPMV_TASKS
#ifdef PRINT_AVG_DEGREE_OF_SPMV_TASKS
      int nDeps = 0;
      for (int task = 0; task < ntasks; ++task) {
        nDeps += fusedSchedule->nparents[task];
      }
      printf("average degree = %f\n", (double)nDeps/ntasks);
#endif

      /*for (int task = 0; task < ntasks; ++task) {
        printf("(");
        for (int j = 0; j < fusedSchedule->nparents[task]; ++j) {
          int parent = fusedSchedule->parents[task][j];
          int t = distance(
            threadBoundaries.begin(),
            upper_bound(threadBoundaries.begin(), threadBoundaries.end(), parent)) - 1;
          printf("%d(%d:%d) ", parent, t, parent - threadBoundaries[t]);
        }
        printf(")->%d\n", task);
      }*/
    } // omp single
  } // construct FusedGSAndSpMVSchedule
#endif // !_OPENMP

#ifdef _OPENMP
#pragma omp barrier
#pragma omp single
#endif
  {

#ifdef TRSOLVER_LOG
  double dt = (__rdtsc() - t)/get_cpu_freq();
  printf(
    "construct task dependency graph takes %f (%f SpMVs)\n", dt, dt/spmvTime);

  printf(
    "average degree = %f\n",
    (double)perThreadRowPtrSum[nthreads]/(levIndices.size() - 2)/nthreads);
  fflush(stdout);
#endif

  t = __rdtsc();

  } // omp single

  // load-balanced partition
  if (tid == nthreads - 1) {
    v1Boundaries2[tid + 1] = ntasks;
  }
  else {
    int nnzPerThread = (perThreadRowPtrSum[nthreads] + nthreads - 1)/nthreads;
    int boundaryTid =
      upper_bound(
        perThreadRowPtrSum, perThreadRowPtrSum + nthreads - 1,
        nnzPerThread*(tid + 1)) -
      perThreadRowPtrSum - 1;

    v1Boundaries2[tid + 1] =
      upper_bound(
        taskAdjacency.rowptr + v1Boundaries[boundaryTid],
        taskAdjacency.rowptr + v1Boundaries[boundaryTid + 1],
        nnzPerThread*(tid + 1) + perThreadOrigRowPtrSum[boundaryTid] - perThreadRowPtrSum[boundaryTid]) -
      taskAdjacency.rowptr - 1;
    v1Boundaries2[tid + 1] = max(0, v1Boundaries2[tid + 1]);
  }

#ifdef _OPENMP
#pragma omp barrier
#pragma omp single
#endif
  {
    for (int i = 0; i < nthreads; ++i) {
    }
  }

  v1Begin = v1Boundaries2[tid];
  v1End = v1Boundaries2[tid + 1];

  t1Begin = distance(
    threadBoundaries.begin(),
    upper_bound(threadBoundaries.begin(), threadBoundaries.end(), v1Begin)) - 1;
  t1End = distance(
    threadBoundaries.begin(),
    upper_bound(threadBoundaries.begin(), threadBoundaries.end(), v1End)) - 1;
  t1End = min(t1End, nthreads - 1);

  rowPtrCnt = 0;
  maxL = 0;
  for (int v1 = v1Begin; v1 < v1End; ++v1) {
    rowPtrCnt += nnzPerRow[v1];
    maxL = max(maxL, nnzPerRow[v1]);
  }
  perThreadRowPtrSum[tid + 1] = rowPtrCnt;

#ifdef _OPENMP
#pragma omp barrier
#pragma omp single
#endif
  {
    for (int tid = 0; tid < nthreads; ++tid ){
      perThreadRowPtrSum[tid + 1] += perThreadRowPtrSum[tid];
    }
  }

  rowPtrCnt = 0;

#ifdef _OPENMP
  if (transitiveReduction)
#endif
  {
    unsigned long long ttt = __rdtsc();

#ifdef _OPENMP
#pragma omp barrier
#pragma omp single
#endif
    {
      childrenBufEnd = memoryPool->getHead();

      childrenBuf = schedule->allocateFront<int>(perThreadRowPtrSum[nthreads]);
      children = schedule->allocateFront<int *>(ntasks);
      numOfChildren = schedule->allocateFront<short>(ntasks);
    }

#ifdef _OPENMP
#pragma omp for
#endif
    for (int i = 0; i < ntasks; ++i) {
      children[i] = NULL;
      numOfChildren[i] = 0;
    }

    t11 = 0;
    t2 = 0;
    vector<int> tempAdj(maxL + 1);
    //int itr1DistanceSum = 0, itr2DistanceSum = 0, itrDistanceCnt = 0;
    //int itr1ContSum = 0, itr1ContCnt = 0, itr2ContSum = 0, itr2ContCnt = 0;
    int skipSum = 0, skipCnt = 0;
    for (int t = t1Begin; t <= t1End; ++t) {
      for (int j = max(threadBoundaries[t], v1Begin); j < min(threadBoundaries[t + 1], v1End); ++j) {
        unsigned long long tempTimer = __rdtsc();

        children[j] = childrenBuf + perThreadRowPtrSum[tid] + rowPtrCnt;
        int *adj = taskAdjacency.colidx + taskAdjacency.rowptr[j];
        int adjSize = nnzPerRow[j];

        int size;
        if (j < threadBoundaries[t + 1] - 1) {
          int idx = lower_bound(adj, adj + adjSize, j + 1) - adj;
          if (idx < adjSize && adj[idx] == j + 1) {
            size = adjSize;
            memcpy(&tempAdj[0], adj, sizeof(adj[0])*adjSize);
          }
          else {
            size = adjSize + 1;
            memcpy(&tempAdj[0], adj, sizeof(adj[0])*idx);
            tempAdj[idx] = j + 1;
            memcpy(&tempAdj[idx + 1], adj + idx, sizeof(adj[0])*(adjSize - idx));
          }
        }
        else {
          size = adjSize;
          memcpy(&tempAdj[0], adj, sizeof(adj[0])*adjSize);
        }

        t11 += __rdtsc() - tempTimer;
        tempTimer = __rdtsc();

        int idx = 0;
        for (int *succItr = adj; succItr != adj + adjSize; ++succItr) {
          int i = *succItr;

          int *itr1 = taskInvAdjacency.colidx + taskInvAdjacency.rowptr[i];
          int *itr1End = itr1 + taskInvAdjacencyLengths[i];
          int *itr2 = &tempAdj[0];
          int *itr2End = itr2 + size;

          bool toRemove = false;

          if (itr2 != itr2End) {
            itr1 = lower_bound(itr1, itr1End, *itr2);
          }
          if (itr1 != itr1End) {
            itr2 = lower_bound(itr2, itr2End, *itr1);

            /*itr1End = upper_bound(itr1, itr1End, *(itr2End - 1));
            itr2End = upper_bound(itr2, itr2End, *(itr1End - 1));*/
          }

          while (itr1 != itr1End && itr2 != itr2End) {
            int cmp = *itr1 - *itr2;
            if (0 == cmp) {
              toRemove = true;
              break;
            }
            else if (cmp < 0) ++itr1;
            else ++itr2;
          }

          if (!toRemove) {
            children[j][idx] = *succItr;
            __sync_fetch_and_add(&nparentsForward[*succItr], 1);
            ++idx;
          }
        }

        numOfChildren[j] = idx;
        rowPtrCnt += idx;

        t2 += __rdtsc() - tempTimer;
      } // for each j
    } // for each t

    if (20 == tid) {
      //printf("skip = %f\n", (double)skipSum/skipCnt);
      /*printf(
        "itr1Distance = %f, itr2Distance = %f, itr1Cont = %f, itr2Cont = %f\n",
        (double)itr1DistanceSum/itrDistanceCnt,
        (double)itr2DistanceSum/itrDistanceCnt,
        (double)itr1ContSum/itr1ContCnt,
        (double)itr2ContSum/itr2ContCnt);*/
    }

    numNewEdges[tid] = rowPtrCnt;
    tempAdj.clear();

    tempTimer = __rdtsc();
#ifdef _OPENMP
#pragma omp barrier
#endif
    t3 = __rdtsc() - tempTimer;
    barrierTimes[tid] = t3;

#ifdef _OPENMP
#pragma omp barrier
#endif
    // convert numOfChildren and children
    // to nparents and parents
    size_t nparentsTempBegin;

#ifdef _OPENMP
#pragma omp master
#endif
    {
      taskAdjacency.~CSR();
      taskInvAdjacency.~CSR();
      if (useMemoryPool) {
        memoryPool->setTail(adjacencyBegin);
      }
      else {
        FREE(taskInvAdjacencyLengths);
      }

      int cForward = 0, cBackward = 0;
      for (int i = 0; i < ntasks; ++i) {
        cForward += nparentsForward[i];
        cBackward += numOfChildren[i];

        nparentsBackward[i] = numOfChildren[i];
      }

      parentsBuf[0] = schedule->allocate<int>(cForward);
      parentsBuf[1] = schedule->allocate<int>(cBackward);

      cForward = 0, cBackward = 0;
      for (int i = 0; i < ntasks; ++i) {
        parentsForward[i] = parentsBuf[0] + cForward;
        cForward += nparentsForward[i];

        parentsBackward[i] = parentsBuf[1] + cBackward;
        cBackward += nparentsBackward[i];
      }

      nparentsTempBegin = memoryPool->getTail();
      nparentsTemp = schedule->allocate<short>(ntasks);
    } // omp single
#ifdef _OPENMP
#pragma omp barrier

#pragma omp for
#endif
    for (int i = 0; i < ntasks; ++i) {
      nparentsTemp[i] = 0;
    }

#ifdef _OPENMP
#pragma omp for
#endif
    for (int i = 0; i < ntasks; ++i) {
      for (int j = 0; j < numOfChildren[i]; ++j) {
        int child = children[i][j];
        int idx = __sync_fetch_and_add(nparentsTemp + child, 1);
        parentsForward[child][idx] = i;
        parentsBackward[i][j] = child;
      }
    }

#ifdef _OPENMP
#pragma omp master
#endif
    {
      if (useMemoryPool) {
        memoryPool->setTail(nparentsTempBegin);
        memoryPool->setHead(childrenBufEnd);
        nparentsTemp = NULL;
        children = NULL;
        numOfChildren = NULL;
        childrenBuf = NULL;
      }
      else {
        FREE(nparentsTemp);
        FREE(children);
        FREE(numOfChildren);
        FREE(childrenBuf);
      }
    } // omp single
#ifdef _OPENMP
#pragma omp barrier
#endif
  } // if (transitiveReduction)
#ifdef _OPENMP
  else {
#pragma omp barrier
#pragma omp single
    {
      int cForward = 0, cBackward = 0;
      for (int i = 0; i < ntasks; ++i) {
        nparentsForward[i] = taskInvAdjacencyLengths[i];
        nparentsBackward[i] = nnzPerRow[i];

        cForward += nparentsForward[i];
        cBackward += nparentsBackward[i];
      }
      FREE(taskInvAdjacencyLengths);

      parentsBuf[0] = schedule->allocate<int>(cForward);
      parentsBuf[1] = schedule->allocate<int>(cBackward);

#ifdef LOG_MEMORY_ALLOCATION
      printf("Allocate %d MB in %s\n", sizeof(int)*cForward/1024/1024, __func__);
#endif

      cForward = 0, cBackward = 0;
      for (int i = 0; i < ntasks; ++i) {
        parentsForward[i] = parentsBuf[0] + cForward;
        cForward += nparentsForward[i];

        parentsBackward[i] = parentsBuf[1] + cBackward;
        cBackward += nparentsBackward[i];

        for (int j = 0; j < nparentsForward[i]; ++j) {
          parentsForward[i][j] = taskInvAdjacency.colidx[taskInvAdjacency.rowptr[i] + j];
        }

        for (int j = 0; j < nparentsBackward[i]; ++j) {
          parentsBackward[i][j] = taskAdjacency.colidx[taskAdjacency.rowptr[i] + j];
        }
      }

      taskAdjacency.~CSR();
      taskInvAdjacency.~CSR();
    } // omp single
  } // !transitiveReduction
#endif
  } // omp parallel

  unsigned long long ttt = __rdtsc();

#ifdef TRSOLVER_LOG
  double dt = (__rdtsc() - t)/get_cpu_freq();
  printf("transitive reduction takes %f (%f SpMVs) \n", dt, dt/spmvTime);

  int numNewEdgesSum = 0;
  for (int i = 0; i < nthreads; ++i) {
    numNewEdgesSum += numNewEdges[i];
  }
  printf(
    "average degree after reduction = %f\n",
    (double)numNewEdgesSum/(levIndices.size() - 2)/nthreads);
#endif

  if (useMemoryPool) {
    memoryPool->setHead(origRowIdToTaskIdEnd);
  }
  else {
    FREE(origRowIdToTaskId);
  }

  /*for (int u = 0; u < ntasks; ++u) {
    int d = taskAdjacency[u].size();
    numOfChildren[u] = d;
    children[u] = (int *)_mm_malloc(sizeof(int)*d, 64);

    auto itr(taskAdjacency[u].begin());
    for (int i = 0; itr != taskAdjacency[u].end(); ++itr, ++i) {
      children[u][i] = *itr;
      nparents[*itr]++;
    }
  }*/

  // count mergeable tasks
  /*int mergeables = 0;
  for (int tid = 0; tid < nthreads; ++tid) {
    for (int task = threadBoundariesForward[tid]; task < threadBoundariesForward[tid + 1] - 1; ++task) {
      if (numOfChildrenForward[task] == 0 && nparentsForward[task + 1] == 0) {
        mergeables++;
      }
    }
  }
  printf("mergables = %f\n", (double)mergeables/threadBoundariesForward[nthreads]);*/

#ifdef TRSOLVER_LOG
  printf("parallelism = %f\n", (double)m/(levIndices.size() - 1));
#endif
}

void LevelSchedule::constructTaskGraph(
  int m,
  const int *rowptr, const int *diagptr, const int *extptr, const int *colidx,
  const CostFunction& costFunction)
{
  if (0 == rowptr[0]) {
    return constructTaskGraph_<0>(this, m, rowptr, diagptr, extptr, colidx, costFunction);
  }
  else {
    assert(1 == rowptr[0]);
    return constructTaskGraph_<1>(this, m, rowptr, diagptr, extptr, colidx, costFunction);
  }
}

} // namespace SpMP
