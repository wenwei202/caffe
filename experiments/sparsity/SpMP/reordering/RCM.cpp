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
 * \brief Parallel BFS and RCM reordering
 *
 * \author Jongsoo Park (jongsoo.park@intel.com)
 *
 * \ref "Parallelization of Reordering Algorithms for Bandwidth and Wavefront
 *       Reduction", Karantasis et al., SC 2014
 * \ref "AN OBJECT-ORIENTED ALGORITHMIC LABORATORY FOR ORDERING SPARSEMATRICES",
 *       Kumfert
 * \ref "Fast and Efficient Graph Traversal Algorithms for CPUs: Maximizing
 *       Single-Node Efficiency", Chhugani et al., IPDPS 2012
 */

#include <cassert>
#include <climits>
#include <cstring>
#include <cstdio>

#include <vector>
#include <algorithm>

#include <omp.h>

#include "../CSR.hpp"
#include "BitVector.hpp"
#include "../synk/barrier.hpp"

using namespace std;

namespace SpMP
{

const int PAR_THR = 65536;

void findConnectedComponents(
  const CSR *A,
  int *numOfComponents, int **compToRoot, int **compSizes, int **compSizePrefixSum,
  int **nodesSortedByComp);

void findConnectedComponentsWithBitVector(
  const CSR *A,
  int *numOfComponents, int **compToRoot, int **compSizes, int **compSizePrefixSum,
  int **nodesSortedByComp,
  const BitVector *bv);

// compute prefix sum of levels
static void prefixSumOfLevels(
  int *prefixSum,
  const CSR *A, const int *levels, int numLevels,
  const int *components, int sizeOfComponents,
  bool parallel = true)
{
  if (parallel) {
    int *local_count = new int[omp_get_max_threads()*numLevels];
    int *local_sum_array = new int[omp_get_max_threads() + 1];

#pragma omp parallel
    {
      int nthreads = omp_get_num_threads();
      int tid = omp_get_thread_num();

      memset(local_count + tid*numLevels, 0, sizeof(int)*numLevels);

      if (NULL == components) {
#pragma omp for
        for (int i = 0; i < A->m; ++i) {
          assert(levels[i] != INT_MAX);
          ++local_count[tid*numLevels + levels[i]];
        }
      }
      else {
#pragma omp for
        for (int i = 0; i < sizeOfComponents; ++i) {
          assert(components[i] >= 0 && components[i] < A->m);
          assert(levels[components[i]] != INT_MAX);
          ++local_count[tid*numLevels + levels[components[i]]];
        }
      }

      int lPerThread = (numLevels + nthreads - 1)/nthreads;
      int lBegin = min(lPerThread*tid, numLevels);
      int lEnd = min(lBegin + lPerThread, numLevels);

      int local_sum = 0;

      for (int l = lBegin; l < lEnd; ++l) {
        prefixSum[l] = local_sum;
        for (int t = 0; t < nthreads; ++t) {
          local_sum += local_count[t*numLevels + l];
        }
      }

      local_sum_array[tid + 1] = local_sum;

#pragma omp barrier
      if (0 == tid)
      {
        for (int t = 1; t < nthreads; ++t) {
          local_sum_array[t + 1] += local_sum_array[t];
        }
        assert(local_sum_array[nthreads] == sizeOfComponents);
        prefixSum[numLevels] = sizeOfComponents;
      }
#pragma omp barrier

      if (tid > 0) {
        local_sum = local_sum_array[tid];
        for (int l = lBegin; l < lEnd; ++l) {
          prefixSum[l] += local_sum;
        }
      }
    } // omp parallel

    delete[] local_count;
    delete[] local_sum_array;
  } // parallel
  else {
    memset(prefixSum, 0, sizeof(int)*(numLevels + 1));

    if (NULL == components) {
      for (int i = 0; i < A->m; ++i) {
        assert(levels[i] != INT_MAX);
        ++prefixSum[levels[i] + 1];
      }
    }
    else {
      for (int i = 0; i < sizeOfComponents; ++i) {
        assert(components[i] >= 0 && components[i] < A->m);
        assert(levels[components[i]] != INT_MAX);
        ++prefixSum[levels[components[i]] + 1];
      }
    }

    for (int l = 0; l < numLevels; ++l) {
      prefixSum[l + 1] += prefixSum[l];
    }
  }
}

struct bfsAuxData
{
  int *q[2];
  int *qTail[2];
  int *qTailPrefixSum;
  int *rowptrs;
  int *nnzPrefixSum;
  int *candidates;

  bfsAuxData(int m);
  ~bfsAuxData();
};

bfsAuxData::bfsAuxData(int m)
{
  q[0] = new int[m*omp_get_max_threads()];
  q[1] = new int[m*omp_get_max_threads()];

  qTail[0] = new int[omp_get_max_threads()];
  qTail[1] = new int[omp_get_max_threads()];

  qTailPrefixSum = new int[omp_get_max_threads() + 1];

  rowptrs = new int[omp_get_max_threads()*m];
  nnzPrefixSum = new int[omp_get_max_threads() + 1];

  candidates = new int[m];
}

bfsAuxData::~bfsAuxData()
{
  delete[] q[0];
  delete[] q[1];
  delete[] qTail[0];
  delete[] qTail[1];
  delete[] qTailPrefixSum;
  delete[] rowptrs;
  delete[] nnzPrefixSum;
  delete[] candidates;
}

/**
 * @return -1 if shortcircuited num of levels otherwise
 *
 * pre-condition: levels should be initialized to -1
 */
template<int BASE = 0, bool OUTPUT_VISITED = false>
int bfs_serial_(
  const CSR *A, int source, int *levels, bfsAuxData *aux,
  int *visited = NULL) {

  int numOfVisited = 0;

  int tid = omp_get_thread_num();

  int numLevels = 0;
  levels[source] = numLevels;

  int **q = aux->q;
  q[0][tid*A->m] = source;

  int qTail[2] = { 1, 0 };

  while (true) {

    if (OUTPUT_VISITED) {
      memcpy(
        visited + numOfVisited, q[numLevels%2] + tid*A->m, qTail[numLevels%2]*sizeof(int));
    }
    numOfVisited += qTail[numLevels%2];

    ++numLevels;
    if (qTail[1 - numLevels%2] == 0) break;

    int *tailPtr = q[numLevels%2] + tid*A->m;

    for (int i = 0; i < qTail[1 - numLevels%2]; ++i) {
      int u = q[1 - numLevels%2][i + tid*A->m];
      assert(levels[u] == numLevels - 1);

      for (int j = A->rowptr[u] - BASE; j < A->rowptr[u + 1] - BASE; ++j) {
        int v = A->colidx[j] - BASE;
        if (-1 == levels[v]) {
          levels[v] = numLevels;

          *tailPtr = v;
          ++tailPtr;
        }
      }
    } // for each current node u

    qTail[numLevels%2] = tailPtr - (q[numLevels%2] + tid*A->m);
  } // while true

  return numLevels;
}

template<bool OUTPUT_VISITED = false>
int bfs_serial(
  const CSR *A, int source, int *levels, bfsAuxData *aux,
  int *visited = NULL)
{
  if (0 == A->getBase()) {
    return bfs_serial_<0, OUTPUT_VISITED>(A, source, levels, aux, visited);
  }
  else {
    assert(1 == A->getBase());
    return bfs_serial_<1, OUTPUT_VISITED>(A, source, levels, aux, visited);
  }
}

/**
 * @return -1 if shortcircuited num of levels otherwise
 *
 * pre-condition: levels should be initialized to -1
 *
 * It implements a part of optimizations (namely bit-vector and atomic-free
 * updates) described in
 * "Fast and Efficient Graph Traversal Algorithms for CPUs: Maximizing
 * Single-Node Efficiency", Chhugani et al., IPDPS 2012
 */
template<int BASE = 0, bool SET_LEVEL = true, bool OUTPUT_VISITED = false>
int bfs_(
  const CSR *A, int source, int *levels, BitVector *bv,
  bfsAuxData *aux,
  int *visited = NULL, int *numOfVisited = NULL,
  int *width = NULL, int *shortCircuitWidth = NULL) {

  int numLevels = 0;
  if (SET_LEVEL) levels[source] = numLevels;
  bv->set(source);

  int **q = aux->q;
  q[0][0] = source;

  int *qTail[2] = { aux->qTail[0], aux->qTail[1] };
  qTail[0][0] = 1;

  int *qTailPrefixSum = aux->qTailPrefixSum;
  qTailPrefixSum[0] = 0;

  int *rowptrs = aux->rowptrs;
  rowptrs[0] = 0;
  rowptrs[1] = A->rowptr[source + 1] - A->rowptr[source];

  int *nnzPrefixSum = aux->nnzPrefixSum;
  nnzPrefixSum[0] = 0;
  nnzPrefixSum[1] = rowptrs[1];

  if (OUTPUT_VISITED) *numOfVisited = 0;

#pragma omp parallel
  {
  int tid = omp_get_thread_num();
  int nthreads = omp_get_num_threads();

  if (tid > 0) {
    qTail[0][tid] = 0;
    nnzPrefixSum[tid + 1] = 0;
  }

  while (true) {
    synk::Barrier::getInstance()->wait(tid);
#pragma omp master
    {
      for (int t = 0; t < nthreads; ++t) {
        qTailPrefixSum[t + 1] = qTailPrefixSum[t] + qTail[numLevels%2][t];
        nnzPrefixSum[t + 1] += nnzPrefixSum[t];
      }
      ++numLevels;
      if (width) {
        *width = max(*width, qTailPrefixSum[nthreads]);
      }
      if (shortCircuitWidth) {
        if (qTailPrefixSum[nthreads] > *shortCircuitWidth) {
          numLevels = -1;
        }
      }
      if (OUTPUT_VISITED) *numOfVisited += qTailPrefixSum[nthreads];
    }
    synk::Barrier::getInstance()->wait(tid);

    if (qTailPrefixSum[nthreads] == 0 || numLevels == -1) break;

    // partition based on # of nnz
    int nnzPerThread = (nnzPrefixSum[nthreads] + nthreads - 1)/nthreads;
    int tBegin = upper_bound(
        nnzPrefixSum, nnzPrefixSum + nthreads + 1,
        nnzPerThread*tid) -
      nnzPrefixSum - 1;
    if (0 == tid) {
      tBegin = 0;
    }
    int tEnd = upper_bound(
        nnzPrefixSum, nnzPrefixSum + nthreads + 1,
        nnzPerThread*(tid + 1)) -
      nnzPrefixSum - 1;
    assert(tBegin >= 0 && tBegin <= nthreads);
    assert(tEnd >= 0 && tEnd <= nthreads);
    assert(tBegin <= tEnd);

    int iBegin, iEnd;
    if (0 == tid) {
      iBegin = 0;
    }
    else if (tBegin == nthreads) {
      iBegin = qTail[1 - numLevels%2][tEnd - 1];
    }
    else {
      iBegin = upper_bound(
          rowptrs + tBegin*A->m, rowptrs + tBegin*A->m + qTail[1 - numLevels%2][tBegin],
          nnzPerThread*tid - nnzPrefixSum[tBegin]) -
        (rowptrs + tBegin*A->m) - 1;
    }

    if (tEnd == nthreads) {
      iEnd = 0;
    }
    else {
      iEnd = upper_bound(
          rowptrs + tEnd*A->m, rowptrs + tEnd*A->m + qTail[1 - numLevels%2][tEnd],
          nnzPerThread*(tid + 1) - nnzPrefixSum[tEnd]) -
        (rowptrs + tEnd*A->m) - 1;
    }

    if (OUTPUT_VISITED) {
      memcpy(
        visited + *numOfVisited - qTailPrefixSum[nthreads] + qTailPrefixSum[tid], 
        q[1 - numLevels%2] + tid*A->m,
        sizeof(int)*(qTailPrefixSum[tid + 1] - qTailPrefixSum[tid]));
    }
    synk::Barrier::getInstance()->wait(tid);

    int *tailPtr = q[numLevels%2] + tid*A->m;
    int *rowptr = rowptrs + tid*A->m;
    *rowptr = 0;

    for (int t = tBegin; t <= tEnd; ++t) {
      for (int i = (t == tBegin ? iBegin : 0);
          i < (t == tEnd ? iEnd : qTail[1 - numLevels%2][t]);
          ++i) {
        int u = q[1 - numLevels%2][t*A->m + i];
        assert(!SET_LEVEL || levels[u] == numLevels - 1);

        for (int j = A->rowptr[u] - BASE; j < A->rowptr[u + 1] - BASE; ++j) {
          int v = A->colidx[j] - BASE;
          if (OUTPUT_VISITED) {
            if (bv->testAndSet(v)) {
              if (SET_LEVEL) levels[v] = numLevels;

              *tailPtr = v;
              *(rowptr + 1) = *rowptr + A->rowptr[v + 1] - A->rowptr[v];

              ++tailPtr;
              ++rowptr;
            }
          }
          else {
            if (!bv->get(v)) {
              bv->set(v);
              if (SET_LEVEL) levels[v] = numLevels;

              *tailPtr = v;
              *(rowptr + 1) = *rowptr + A->rowptr[v + 1] - A->rowptr[v];

              ++tailPtr;
              ++rowptr;
            }
          }
        }
      } // for each current node u
    }

    qTail[numLevels%2][tid] = tailPtr - (q[numLevels%2] + tid*A->m);
    nnzPrefixSum[tid + 1] = *rowptr;
  } // while true
  } // omp parallel

#ifndef NDEBUG
  if (OUTPUT_VISITED) {
    int *temp = new int[*numOfVisited];
    copy(visited, visited + *numOfVisited, temp);
    sort(temp, temp + *numOfVisited);
    for (int i = 0; i < *numOfVisited; ++i) {
      assert(temp[i] >= 0 && temp[i] < A->m);
      if (i > 0 && temp[i] == temp[i - 1]) {
        printf("%d duplicated\n", temp[i]);
        assert(false);
      }
    }

    delete[] temp;
  }
#endif

  return numLevels;
}

template<bool SET_LEVEL = true, bool OUTPUT_VISITED = false>
int bfs(
  const CSR *A, int source, int *levels, BitVector *bv,
  bfsAuxData *aux,
  int *visited = NULL, int *numOfVisited = NULL,
  int *width = NULL, int *shortCircuitWidth = NULL)
{
  if (0 == A->getBase()) {
    return bfs_<0, SET_LEVEL, OUTPUT_VISITED>(
      A, source, levels, bv, aux, visited, numOfVisited, width, shortCircuitWidth);
  }
  else {
    assert(1 == A->getBase());
    return bfs_<1, SET_LEVEL, OUTPUT_VISITED>(
      A, source, levels, bv, aux, visited, numOfVisited, width, shortCircuitWidth);
  }
}

/**
 * Find minimum degree node among unvisited nodes.
 * Unvisited nodes are specified by color array
 */
static int getMinDegreeNode(const CSR *A, const int *nodes, int numOfNodes, bool parallel = true)
{
  int global_min_idx;

  if (parallel) {
    int local_min[omp_get_max_threads()];
    int local_min_idx[omp_get_max_threads()];

#pragma omp parallel
    {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    int temp_min = INT_MAX;
    int temp_min_idx = -1;

#pragma omp for
    for (int i = 0; i < numOfNodes; ++i) {
      int u = nodes[i];
      int degree = A->rowptr[u + 1] - A->rowptr[u];
      if (degree < temp_min) {
        temp_min = degree;
        temp_min_idx = u;
      }
    }

    local_min[tid] = temp_min;
    local_min_idx[tid] = temp_min_idx;

#pragma omp barrier
#pragma omp master
    {
      int global_min = INT_MAX;
      global_min_idx = -1;
      for (int i = 0; i < nthreads; ++i) {
        if (local_min[i] < global_min) {
          global_min = local_min[i];
          global_min_idx = local_min_idx[i];
        }
      }
      //printf("global_min = %d\n", global_min);
    }
    } // omp parallel
  }
  else {
    int global_min = INT_MAX;
    global_min_idx = -1;

    for (int i = 0; i < numOfNodes; ++i) {
      int u = nodes[i];
      int degree = A->rowptr[u + 1] - A->rowptr[u];
      if (degree < global_min) {
        global_min = degree;
        global_min_idx = u;
      }
    }
  }

  return global_min_idx;
}

static void initializeBitVector(
  BitVector *bv, int m, const int *nodes, int numOfNodes)
{
#pragma omp parallel for
  for (int i = 0; i < numOfNodes; ++i) {
    int u = nodes[i];
    assert(u >= 0 && u < m);
    bv->atomicClear(u);
  }
}

/**
 * Implementation of pseudo diameter heuristic described in
 * "AN OBJECT-ORIENTED ALGORITHMIC LABORATORY FOR ORDERING SPARSEMATRICES",
 * by Kumfert
 */
int selectSourcesWithPseudoDiameter(
  const CSR *A, BitVector *bv, const int *components, int sizeOfComponents, bfsAuxData *aux)
{
  // find the min degree node of this connected component
  int s = getMinDegreeNode(A, components, sizeOfComponents);
//#define PRINT_DBG
  int e = -1;

  int tid = omp_get_thread_num();
  int *candidates = aux->candidates;

  bool first = true;
  while (e == -1) {
    if (!first) initializeBitVector(bv, A->m, components, sizeOfComponents);
    first = false;
    int diameter = bfs<false>(A, s, NULL, bv, aux);

    int nCandidates = 0;
    for (int t = 0; t < omp_get_max_threads(); ++t) {
      for (int j = 0; j < aux->qTail[diameter%2][t]; ++j) {
        int u = aux->q[diameter%2][t*A->m + j];
        candidates[nCandidates] = u;
        ++nCandidates;
      }
    }

    // sort by vertex by ascending degree
    for (int i = 1; i < nCandidates; ++i) {
      int c = candidates[i];
      int degree = A->rowptr[c + 1] - A->rowptr[c];

      int j = i - 1;
      while (j >= 0 && A->rowptr[candidates[j] + 1] - A->rowptr[candidates[j]] > degree) {
        candidates[j + 1] = candidates[j];
        --j;
      }

      candidates[j + 1] = c;
    }

    // select first 5 that are not adjacent to any previously chosen vertex
    int outIdx = 1;
    for (int i = 1; i < nCandidates && outIdx < 5; ++i) {
      int u = candidates[i];
      bool adjacent = false;
      for (int k = 0; !adjacent && k < outIdx; ++k) {
        if (candidates[k] == u) adjacent = true;
      }
      for (int j = A->rowptr[u]; !adjacent && j < A->rowptr[u + 1]; ++j) {
        int v = A->colidx[j];
        for (int k = 0; !adjacent && k < outIdx; ++k) {
          if (candidates[k] == v) adjacent = true;
        }
      }
      if (!adjacent) {
        candidates[outIdx++] = u;
      }
    }

    nCandidates = outIdx;

    int minWidth = INT_MAX;
    for (int i = 0; i < nCandidates; ++i) {
      initializeBitVector(bv, A->m, components, sizeOfComponents);

      int width = INT_MIN;
      int newDiameter =
        bfs<false>(
          A, candidates[i], NULL, bv, aux, NULL, NULL, &width, &minWidth);

      if (-1 == newDiameter) { // short circuited
        continue;
      }
      else if (newDiameter > diameter && width < minWidth) {
        s = candidates[i];
        e = -1;
        break;
      }
      else if (width < minWidth) {
        minWidth = width;
        e = candidates[i];
      }
    }
  } // iterate to find maximal diameter

  return s;
}

class DegreeComparator
{
public :
  DegreeComparator(const int *rowptr) : rowptr_(rowptr) { };

  bool operator()(int a, int b) {
    return rowptr_[a + 1] - rowptr_[a] < rowptr_[b + 1] - rowptr_[b];
  }

private :
  const int *rowptr_;
};

void CSR::getRCMPermutation(int *perm, int *inversePerm, bool pseudoDiameterSourceSelection /*= true*/)
{
  if (m != n) {
    fprintf(stderr, "BFS permutation only supports square matrices\n");
    exit(-1);
  }
  assert(isSymmetric(false)); // check structural symmetry

  int oldBase = getBase();
  make0BasedIndexing();

  // 1. Start vertex
  double bfsTime1 = 0, prefixTime1 = 0, placeTime1 = 0;
  double sourceSelectionTime2 = 0, bfsTime2 = 0, prefixTime2 = 0, placeTime2 = 0;

  int *levels = new int[m];
  int maxDegree = INT_MIN;
#pragma omp parallel for reduction(max:maxDegree)
  for (int i = 0; i < m; ++i) {
    maxDegree = max(maxDegree, rowptr[i + 1] - rowptr[i]);
  }
#ifdef PRINT_DBG
  printf("maxDegree = %d\n", maxDegree);
#endif

  BitVector bv(m);

  int *children_array = new int[omp_get_max_threads()*max(PAR_THR, maxDegree)];

  int singletonCnt = 0, twinCnt = 0;

  bfsAuxData aux(m);
  volatile int *write_offset = new int[(m + 1)*16];
  int *prefixSum_array = new int[max(omp_get_max_threads()*PAR_THR, m) + 1];

  DegreeComparator comparator(rowptr);

  int numOfComponents;
  int *compToRoot, *compSizes, *compSizePrefixSum;
  int *nodesSortedByComp;

  double timeConnectedComponents = -omp_get_wtime();
  findConnectedComponents(
    this,
    &numOfComponents, &compToRoot, &compSizes, &compSizePrefixSum,
    &nodesSortedByComp);
  timeConnectedComponents += omp_get_wtime();

//#define MEASURE_LOAD_BALANCE
  double tBegin = omp_get_wtime();
#ifdef MEASURE_LOAD_BALANCE
  double barrierTimes[omp_get_max_threads()];
  double barrierTimeSum = 0;
#endif

#pragma omp parallel reduction(+:singletonCnt,twinCnt,bfsTime1,prefixTime1,placeTime1)
  {
    int tid = omp_get_thread_num();

    int *children = children_array + tid*PAR_THR;
    int *prefixSum = prefixSum_array + tid*(PAR_THR + 1);

  // for each small connected component
#pragma omp for
  for (int c = 0; c < numOfComponents; ++c) {
    int i = compToRoot[c];
    int offset = compSizePrefixSum[c];

    // 1. Short circuiting
    if (compSizes[c] >= PAR_THR) continue;

    // short circuit for a singleton or a twin
    if (rowptr[i + 1] == rowptr[i] + 1 && colidx[rowptr[i]] == i || rowptr[i + 1] == rowptr[i]) {
      inversePerm[m - offset - 1] = i;
      perm[i] = m - offset - 1;
      ++singletonCnt;
      continue;
    }
    else if (rowptr[i + 1] == rowptr[i] + 1) {
      int u = colidx[rowptr[i]];
      if (rowptr[u + 1] == rowptr[u] + 1 && colidx[rowptr[u]] == i) {
        inversePerm[m - offset - 1] = i;
        inversePerm[m - (offset + 1) - 1] = u;
        perm[i] = m - offset - 1;
        perm[u] = m - (offset + 1) - 1;
        ++twinCnt;
        continue;
      }
    }
    else if (rowptr[i + 1] == rowptr[i] + 2) {
      int u = -1;
      if (colidx[rowptr[i]] == i) {
        u = colidx[rowptr[i] + 1];
      }
      else if (colidx[rowptr[i] + 1] == i) {
        u = colidx[rowptr[i]];
      }
      if (u != -1 &&
        rowptr[u + 1] == rowptr[u] + 2 &&
          (colidx[rowptr[u]] == u && colidx[rowptr[u] + 1] == i ||
            colidx[rowptr[u] + 1] == u && colidx[rowptr[u]] == i)) {
        inversePerm[m - offset - 1] = i;
        inversePerm[m - (offset + 1) - 1] = u;
        perm[i] = m - offset - 1;
        perm[u] = m - (offset + 1) - 1;
        ++twinCnt;
        continue;
      }
    }

    // collect nodes of this connected component
    int *components = nodesSortedByComp + compSizePrefixSum[c];

    // 2. BFS
    double t = omp_get_wtime();
    for (int i = 0; i < compSizes[c]; ++i) {
      levels[components[i]] = -1;
    }
    int numLevels = bfs_serial(this, i, levels, &aux);
    bfsTime1 += omp_get_wtime() - t;

    // 3. Reorder
    t = omp_get_wtime();
    prefixSumOfLevels(
      prefixSum, this, levels, numLevels, components, compSizes[c], false);
    prefixTime1 += omp_get_wtime() - t;

    t = omp_get_wtime();
    inversePerm[m - offset - 1] = i;
    perm[i] = m - offset - 1;

    for (int l = 0; l < numLevels; ++l) {
      int r = prefixSum[l] + offset;
      int w = prefixSum[l + 1] + offset;
      while (r != prefixSum[l + 1] + offset) {
        int u = inversePerm[m - r - 1];
        ++r;
        int childrenIdx = 0;
        for (int j = rowptr[u]; j < rowptr[u + 1]; ++j) {
          int v = colidx[j];
          if (levels[v] == l + 1) {
            children[childrenIdx] = v;
            ++childrenIdx;
            levels[v] = -1;
          }
        }

        std::sort(children, children + childrenIdx, comparator);

        for (int i = 0; i < childrenIdx; ++i) {
          int c = children[i];
          int idx = m - (w + i) - 1;
          inversePerm[idx] = c;
          perm[c] = idx;
        }
        w += childrenIdx;
      }
    }

    placeTime1 += omp_get_wtime() - t;
  }

#ifdef MEASURE_LOAD_BALANCE
    double t = omp_get_wtime();
#pragma omp barrier
    barrierTimes[tid] = omp_get_wtime() - t;

#pragma omp master
    {
      double tEnd = omp_get_wtime();
      for (int i = 0; i < omp_get_num_threads(); ++i) {
        barrierTimeSum += barrierTimes[i];
      }
      printf("%f load imbalance = %f\n", tEnd - tBegin, barrierTimeSum/(tEnd - tBegin)/omp_get_num_threads());
    }
#undef MEASURE_LOAD_BALANCE
#endif // MEASURE_LOAD_BALANCE
  } // omp parallel

  double timeFirstPhase = omp_get_wtime() - tBegin;
  tBegin = omp_get_wtime();

  int *prefixSum = prefixSum_array;

  // for each large connected component
  int largeCompCnt = 0;
  for (int c = 0; c < numOfComponents; ++c) {
    int i = compToRoot[c];
    int offset = compSizePrefixSum[c];

    // 1. Automatic selection of source
    if (compSizes[c] < PAR_THR) continue;
    ++largeCompCnt;

    double t = omp_get_wtime();

    int *components = nodesSortedByComp + compSizePrefixSum[c];

    // select source
    int source = pseudoDiameterSourceSelection
      ? selectSourcesWithPseudoDiameter(
        this, &bv, components, compSizes[c], &aux)
      : i;
    assert(source >= 0 && source < m);

    sourceSelectionTime2 += omp_get_wtime() - t;

    // 2. BFS
    t = omp_get_wtime();
    initializeBitVector(&bv, m, components, compSizes[c]);
    int numLevels = bfs(this, source, levels, &bv, &aux);
#ifdef PRINT_DBG
    printf("numLevels = %d\n", numLevels);
#endif
    bfsTime2 += omp_get_wtime() - t;

    // 3. Reorder
    t = omp_get_wtime();
    prefixSumOfLevels(
      prefixSum, this, levels, numLevels, components, compSizes[c]);
    prefixTime2 += omp_get_wtime() - t;

    t = omp_get_wtime();
    inversePerm[m - offset - 1] = source;
    perm[source] = m - offset - 1;

    /**
     * Implementation of place function in Algorithm 6 in
     * "Parallelization of Reordering Algorithms for Bandwidth and Wavefront
     * Reduction", Karantasis et al., SC 2014
     */
#pragma omp parallel
    {
      int nthreads = omp_get_num_threads();
      int tid = omp_get_thread_num();

      int *children = children_array + tid*maxDegree;

      for (int l = tid; l <= numLevels; l += nthreads) {
        write_offset[16*l] = prefixSum[l] + offset;
      }
      if (0 == tid) {
        write_offset[0] = offset + 1;
      }

#pragma omp barrier

      for (int l = tid; l < numLevels; l += nthreads) {
        int r = prefixSum[l] + offset;
        while (r != prefixSum[l + 1] + offset) {
          while (r == write_offset[16*l]); // spin
          int u = inversePerm[m - r - 1];
          ++r;
          int childrenIdx = 0;
          for (int j = rowptr[u]; j < rowptr[u + 1]; ++j) {
            int v = colidx[j];
            if (levels[v] == l + 1) {
              children[childrenIdx] = v;
              ++childrenIdx;
              levels[v] = -1;
            }
          }

          std::sort(children, children + childrenIdx, comparator);

          int w = write_offset[16*(l + 1)];
          for (int i = 0; i < childrenIdx; ++i) {
            int c = children[i];
            int idx = m - (w + i) - 1;
            inversePerm[idx] = c;
            perm[c] = idx;
            write_offset[16*(l + 1)] = w + i + 1;
          }
        }
      } // for each level
    } // omp parallel

    placeTime2 += omp_get_wtime() - t;
  }

  double timeSecondPhase = omp_get_wtime() - tBegin;

  delete[] levels;
  delete[] children_array;
  delete[] write_offset;
  delete[] prefixSum_array;

#if 0
  printf("num of connected components = %d (singleton = %d, twin = %d, large (>=%d) = %d)\n", numOfComponents, singletonCnt, twinCnt, PAR_THR, largeCompCnt);
  printf("connectedComponentTime = %f\n", timeConnectedComponents);
  printf("firstPhaseTime (parallel over components) = %f\n", timeFirstPhase);
  printf("\tbfsTime = %f\n", bfsTime1/omp_get_max_threads());
  printf("\tprefixTime = %f\n", prefixTime1/omp_get_max_threads());
  printf("\tplaceTime = %f\n", placeTime1/omp_get_max_threads());
#ifdef MEASURE_LOAD_BALANCE
  printf("\tloadImbalanceTime = %f\n", barrierTimeSum/omp_get_max_threads());
#endif
  printf("secondPhaseTime (parallel within components) = %f\n", timeSecondPhase);
  printf("\tsourceSelectionTime = %f\n", sourceSelectionTime2);
  printf("\tbfsTime = %f\n", bfsTime2);
  printf("\tprefixTime = %f\n", prefixTime2);
  printf("\tplaceTime = %f\n", placeTime2);
#endif

  if (1 == oldBase) {
    make1BasedIndexing();
  }
}

void CSR::getBFSPermutation(int *perm, int *inversePerm)
{
  if (m != n) {
    fprintf(stderr, "BFS permutation only supports square matrices\n");
    exit(-1);
  }
  assert(isSymmetric(false)); // check structural symmetry

  BitVector bv(m);

  bfsAuxData aux(m);

  // First run bfs from node 0 to optimize for the case with a single connected
  // component.
  double timeSecondPhase = -omp_get_wtime();
  int nNodesinFirstComp = 0;
  int numLevels = bfs<false, true>(this, 0, NULL, &bv, &aux, inversePerm, &nNodesinFirstComp);
#ifdef PRINT_DBG
  printf("numLevels = %d\n", numLevels);
#endif
#pragma omp parallel for
  for (int i = 0; i < nNodesinFirstComp; ++i) {
    perm[inversePerm[i]] = i;
  }
  timeSecondPhase += omp_get_wtime();
  if (nNodesinFirstComp == m) {
    return;
  }

  int *levels = new int[m];

  int numOfComponents;
  int *compToRoot, *compSizes, *compSizePrefixSum;
  int *nodesSortedByComp;

  double timeConnectedComponents = -omp_get_wtime();
  findConnectedComponentsWithBitVector(
    this,
    &numOfComponents, &compToRoot, &compSizes, &compSizePrefixSum,
    &nodesSortedByComp,
    &bv);
  timeConnectedComponents += omp_get_wtime();

  int singletonCnt = 0, twinCnt = 0;

  double timeFirstPhase = -omp_get_wtime();

  int base = getBase();

  // for each small connected component
#pragma omp for
  for (int c = 0; c < numOfComponents; ++c) {
    int i = compToRoot[c];
    int offset = compSizePrefixSum[c] + nNodesinFirstComp;

    if (compSizes[c] >= PAR_THR || bv.get(i)) continue;

    // short circuit for a singleton or a twin
    if (rowptr[i + 1] == rowptr[i] + 1 && colidx[rowptr[i] - base] - base == i || rowptr[i + 1] == rowptr[i]) {
      inversePerm[offset] = i;
      perm[i] = offset;
      ++singletonCnt;
      continue;
    }
    else if (rowptr[i + 1] == rowptr[i] + 1) {
      int u = colidx[rowptr[i] - base] - base;
      if (rowptr[u + 1] == rowptr[u] + 1 && colidx[rowptr[u] - base] - base == i) {
        inversePerm[offset] = i;
        inversePerm[offset + 1] = u;
        perm[i] = offset;
        perm[u] = offset + 1;
        ++twinCnt;
        continue;
      }
    }
    else if (rowptr[i + 1] == rowptr[i] + 2) {
      int u = -1;
      if (colidx[rowptr[i] - base] - base == i) {
        u = colidx[rowptr[i] + 1 - base] - base;
      }
      else if (colidx[rowptr[i] + 1 - base] - base == i) {
        u = colidx[rowptr[i] - base] - base;
      }
      if (u != -1 &&
        rowptr[u + 1] == rowptr[u] + 2 &&
          (colidx[rowptr[u] - base] - base == u && colidx[rowptr[u] + 1 - base] - base == i ||
            colidx[rowptr[u] + 1 - base] - base == u && colidx[rowptr[u] - base] - base == i)) {
        inversePerm[offset] = i;
        inversePerm[offset + 1] = u;
        perm[i] = offset;
        perm[u] = offset + 1;
        ++twinCnt;
        continue;
      }
    }

    int *components = nodesSortedByComp + compSizePrefixSum[c];
    for (int i = 0; i < compSizes[c]; ++i) {
      levels[components[i]] = -1;
    }
    bfs_serial<true>(this, i, levels, &aux, inversePerm + offset);
    for (int i = 0; i < compSizes[c]; ++i) {
      perm[inversePerm[offset + i]] = offset + i;
    }
  } // for each small connected component

  delete[] levels;

  timeFirstPhase += omp_get_wtime();
  timeSecondPhase -= omp_get_wtime();

  // for each large connected component
  int largeCompCnt = 0;
  for (int c = 0; c < numOfComponents; ++c) {
    int i = compToRoot[c];
    int offset = compSizePrefixSum[c] + nNodesinFirstComp;

    if (compSizes[c] < PAR_THR || bv.get(i)) continue;
    ++largeCompCnt;

    int *components = nodesSortedByComp + compSizePrefixSum[c];

    int temp = -1;
    int numLevels = bfs<false, true>(this, i, NULL, &bv, &aux, inversePerm + offset, &temp);
#ifdef PRINT_DBG
    printf("numLevels = %d\n", numLevels);
#endif
#pragma omp parallel for
    for (int i = 0; i < compSizes[c]; ++i) {
      perm[inversePerm[offset + i]] = offset + i;
    }
  }

  timeSecondPhase += omp_get_wtime();

#if 0
  printf("num of connected components = %d (singleton = %d, twin = %d, large (>=%d) = %d)\n", numOfComponents, singletonCnt, twinCnt, PAR_THR, largeCompCnt);
  printf("connectedComponentTime = %f\n", timeConnectedComponents);
  printf("firstPhaseTime (parallel over components) = %f\n", timeFirstPhase);
  printf("secondPhaseTime (parallel within components) = %f\n", timeSecondPhase);
#endif
}

} // namespace SpMP
