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
 * \brief Parallel connected component detection
 *
 * \author Jongsoo Park (jongsoo.park@intel.com)
 *
 * \ref "Multi-core spanning forest algorithms using the disjoint-set data
 *       structure", Patwary et al., IPDPS 2012
 */

#include <cassert>
#include <cstring>
#include <algorithm>

#include <omp.h>

#include "../CSR.hpp"
#include "BitVector.hpp"

using namespace std;

namespace SpMP
{

/**
 * idx = idx2*dim1 + idx1
 * -> ret = idx1*dim2 + idx2
 *        = (idx%dim1)*dim2 + idx/dim1
 */
static inline int transpose_idx(int idx, int dim1, int dim2)
{
  return idx%dim1*dim2 + idx/dim1;
}

template<int BASE = 0, bool WITH_BIT_VECTOR = false>
void findConnectedComponents_(
  const CSR *A,
  int *numOfComponents, int **compToRoot, int **compSizes, int **compSizePrefixSum,
  int **nodesSortedByComp,
  const BitVector *bv = NULL) // find component only if bv is false
{
  volatile int *p = new int[A->m];

  int cnts[omp_get_max_threads() + 1]; // prefix sum of # of connected components
  cnts[0] = 0;

  *compToRoot = NULL;
  int *rootToComp = new int[A->m];
  *compSizes = NULL;
  *numOfComponents = 0;
  int nComp;
  *nodesSortedByComp = new int[A->m];

  double t = omp_get_wtime();

  int *nodesToFind = NULL;
  int m = A->m;
  if (WITH_BIT_VECTOR) {
    int *nodesToFindArray = new int[A->m*omp_get_max_threads()];
    int nodesToFindCnt[omp_get_max_threads() + 1];
    nodesToFindCnt[0] = 0;

#pragma omp parallel
    {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    int iPerThread = (A->m + nthreads - 1)/nthreads;
    int iBegin = min(iPerThread*tid, A->m);
    int iEnd = min(iBegin + iPerThread, A->m);

    int localCnt = 0;
    for (int i = iBegin; i < iEnd; ++i) {
      if (!bv->get(i)) {
        nodesToFindArray[A->m*tid + localCnt] = i;
        ++localCnt;
      }
    }
    nodesToFindCnt[tid + 1] = localCnt;

#pragma omp barrier
#pragma omp master
    {
      for (int i = 1; i < nthreads; ++i) {
        nodesToFindCnt[i + 1] += nodesToFindCnt[i];
      }
      m = nodesToFindCnt[nthreads];
      nodesToFind = new int[m];
    }
#pragma omp barrier

    memcpy(
      nodesToFind + nodesToFindCnt[tid], nodesToFindArray + A->m*tid,
      localCnt*sizeof(int));
    } // omp parallel

    delete[] nodesToFindArray;
  } // WITH_BIT_VECTOR

#pragma omp parallel
  {
  int tid = omp_get_thread_num();
  int nthreads = omp_get_num_threads();

  int iPerThread = (m + nthreads - 1)/nthreads;
  int iBegin = min(iPerThread*tid, m);
  int iEnd = min(iBegin + iPerThread, m);

  for (int i = iBegin; i< iEnd; ++i) {
    int ii = WITH_BIT_VECTOR ? nodesToFind[i] : i;
    p[ii] = ii;
  }

#pragma omp barrier

  int xBegin, xEnd;
  if (WITH_BIT_VECTOR) {
    xBegin = iBegin;
    xEnd = iEnd;
  }
  else {
    getLoadBalancedPartition(&xBegin, &xEnd, A->rowptr, A->m);
  }
  assert(xBegin <= xEnd);
  assert(xBegin >= 0 && xBegin <= A->m);
  assert(xEnd >= 0 && xEnd <= A->m);

  for (int x = xBegin; x < xEnd; ++x) {
    int xx = WITH_BIT_VECTOR ? nodesToFind[x] : x;
    for (int j = A->rowptr[xx] - BASE; j < A->rowptr[xx + 1] - BASE; ++j) {
      int y = A->colidx[j] - BASE;
      assert(!WITH_BIT_VECTOR || !bv->get(y));
      if (p[xx] != p[y]) {
        // union
        int r_x = xx, r_y = y;
        while (true) {
          int old_p_r_x = p[r_x]; int old_p_r_y = p[r_y];
          if (old_p_r_x == old_p_r_y) break;

          int old_r_x = r_x; int old_r_y = r_y;

          r_x = old_p_r_x > old_p_r_y ? old_r_x : old_r_y;
          r_y = old_p_r_x > old_p_r_y ? old_r_y : old_r_x;
          int p_r_x = old_p_r_x > old_p_r_y ? old_p_r_x : old_p_r_y;
          int p_r_y = old_p_r_x > old_p_r_y ? old_p_r_y : old_p_r_x;

          if (p_r_x == r_x && __sync_bool_compare_and_swap(&p[r_x], r_x, p_r_y)) {
            break;
          }
          p[r_x] = p_r_y;
          r_x = p_r_x;
        } // while
      } // p[xx] != p[y]
    }
  } // for each row x

#pragma omp barrier

  // path compression so that all p[i] points to its root
  // and count # of components
  int compId = 0;
  for (int i = iBegin; i < iEnd; ++i) {
    int ii = WITH_BIT_VECTOR ? nodesToFind[i] : i;
    int r = ii;
    while (p[r] != r) {
      r = p[r];
      assert(!WITH_BIT_VECTOR || !bv->get(r));
    }
    p[ii] = r;
    if (r == ii) ++compId;
  }

  cnts[tid + 1] = compId;

  // prefix sum # of components
#pragma omp barrier
#pragma omp master
  {
    for (int i = 1; i < nthreads; ++i) {
      cnts[i + 1] += cnts[i];
    }
    *numOfComponents = nComp = cnts[nthreads];
    *compToRoot = new int[nComp];
    *compSizes = new int[nComp];
    *compSizePrefixSum = new int[(nComp + 1)*nthreads];
  }
#pragma omp barrier

  // compId <-> root map
  compId = cnts[tid];
  for (int i = iBegin; i < iEnd; ++i) {
    int ii = WITH_BIT_VECTOR ? nodesToFind[i] : i;
    int r = p[ii];
    if (r == ii) {
      (*compToRoot)[compId] = r;
      rootToComp[r] = compId;
      ++compId;
    }
  }
  
#pragma omp barrier

  // count thread-private component sizes
  int *localPrefixSum = (*compSizePrefixSum) + nComp*tid;
  for (int c = 0; c < nComp; ++c) {
    localPrefixSum[c] = 0;
  }

  for (int i = iBegin; i < iEnd; ++i) {
    int ii = WITH_BIT_VECTOR ? nodesToFind[i] : i;
    int c = rootToComp[p[ii]];
    ++localPrefixSum[c];
  }

#pragma omp barrier

  for (int i = nComp*tid + 1; i < nComp*(tid + 1); ++i) {
    int transpose_i = transpose_idx(i, nthreads, nComp);
    int transpose_i_minus_1 = transpose_idx(i - 1, nthreads, nComp);

    (*compSizePrefixSum)[transpose_i] += (*compSizePrefixSum)[transpose_i_minus_1];
  }

#pragma omp barrier
#pragma omp master
  {
    for (int i = 1; i < nthreads; ++i) {
      int j0 = nComp*i - 1, j1 = nComp*(i + 1) - 1;
      int transpose_j0 = transpose_idx(j0, nthreads, nComp);
      int transpose_j1 = transpose_idx(j1, nthreads, nComp);

      (*compSizePrefixSum)[transpose_j1] += (*compSizePrefixSum)[transpose_j0];
    }
  }
#pragma omp barrier

  if (tid > 0) {
    int transpose_i0 = transpose_idx(nComp*tid - 1, nthreads, nComp);
    
    for (int i = nComp*tid; i < nComp*(tid + 1) - 1; ++i) {
      int transpose_i = transpose_idx(i, nthreads, nComp);

      (*compSizePrefixSum)[transpose_i] += (*compSizePrefixSum)[transpose_i0];
    }
  }

#pragma omp barrier

  int cPerThread = (nComp + nthreads - 1)/nthreads;
  int cBegin = max(min(cPerThread*tid, nComp), 1);
  int cEnd = min(cBegin + cPerThread, nComp);
  if (0 == tid) {
    (*compSizes)[0] = (*compSizePrefixSum)[nComp*(nthreads - 1)];
  }
  for (int c = cBegin; c < cEnd; ++c) {
    (*compSizes)[c] =
      (*compSizePrefixSum)[c + nComp*(nthreads - 1)] -
      (*compSizePrefixSum)[c - 1 + nComp*(nthreads - 1)];
  }

#pragma omp barrier
  
  for (int i = iEnd - 1; i >= iBegin; --i) {
    int ii = WITH_BIT_VECTOR ? nodesToFind[i] : i;
    int c = rootToComp[p[ii]];
    --(*compSizePrefixSum)[c + nComp*tid];
    int offset = (*compSizePrefixSum)[c + nComp*tid];
    (*nodesSortedByComp)[offset] = ii;
  }
  } // omp parallel

  //printf("finding connected components takes %f\n", omp_get_wtime() - t);

#ifndef NDEBUG
  int cnt = 0;
  for (int i = 0; i < nComp; ++i) {
    cnt += (*compSizes)[i];
  }
  assert(cnt == m);

  for (int i = 0; i < nComp; ++i) {
    if (i < nComp - 1)
      assert((*compSizePrefixSum)[i + 1] - (*compSizePrefixSum)[i] == (*compSizes)[i]);

    if ((*compSizes)[i] > 0) {
      // root of each component has the smallest id
      assert((*compToRoot)[i] == (*nodesSortedByComp)[(*compSizePrefixSum)[i]]);
    }

    for (int j = (*compSizePrefixSum)[i]; j < (*compSizePrefixSum)[i] + (*compSizes)[i]; ++j) {
      assert(p[(*nodesSortedByComp)[j]] == (*compToRoot)[i]);
      if (j < (*compSizePrefixSum)[i + 1] - 1) {
        assert((*nodesSortedByComp)[j] < (*nodesSortedByComp)[j + 1]);
      }
    }
  }
#endif

  //printf("num of connected components = %d\n", (*numOfComponents));

  delete[] rootToComp;
  if (nodesToFind) delete[] nodesToFind;
}

void findConnectedComponents(
  const CSR *A,
  int *numOfComponents, int **compToRoot, int **compSizes, int **compSizePrefixSum,
  int **nodesSortedByComp)
{
  if (0 == A->getBase()) {
    findConnectedComponents_<0>(A, numOfComponents, compToRoot, compSizes, compSizePrefixSum, nodesSortedByComp);
  }
  else {
    assert(1 == A->getBase());
    findConnectedComponents_<1>(A, numOfComponents, compToRoot, compSizes, compSizePrefixSum, nodesSortedByComp);
  }
}

void findConnectedComponentsWithBitVector(
  const CSR *A,
  int *numOfComponents, int **compToRoot, int **compSizes, int **compSizePrefixSum,
  int **nodesSortedByComp,
  const BitVector *bv)
{
  if (0 == A->getBase()) {
    findConnectedComponents_<0, true>(A, numOfComponents, compToRoot, compSizes, compSizePrefixSum, nodesSortedByComp, bv);
  }
  else {
    assert(1 == A->getBase());
    findConnectedComponents_<1, true>(A, numOfComponents, compToRoot, compSizes, compSizePrefixSum, nodesSortedByComp, bv);
  }
}

}; // SpMP
