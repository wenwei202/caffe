#include <cstring>
#include <algorithm>

#include "../reordering/BitVector.hpp"
#include "../synk/barrier.hpp"
#include "BFSBipartite.hpp"

using namespace std;
using namespace SpMP;

#if 0
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
  const CSR *A, const CSR *AT,
  int *numOfComponents, int **compToRoot, int **compSizes, int **compSizePrefixSum,
  int **nodesSortedByComp,
  const BitVector *bv = NULL) // find component only if bv is false
{
  volatile int *p = new int[A->m + A->n];

  int cnts[omp_get_max_threads() + 1]; // prefix sum of # of connected components
  cnts[0] = 0;

  *compToRoot = NULL;
  int *rootToComp = new int[A->m + A->n];
  *compSizes = NULL;
  *numOfComponents = 0;
  int nComp;
  *nodesSortedByComp = new int[A->m + A->n];

  double t = omp_get_wtime();

  int *rowsToFind, *colsToFind = NULL;
  int m = A->m;
  int n = A->n;

  if (WITH_BIT_VECTOR) {
    int *rowsToFindArray = new int[A->m*omp_get_max_threads()];
    int *colsToFindArray = new int[A->n*omp_get_max_threads()];
    int rowsToFindCnt[omp_get_max_threads() + 1], colsToFindCnt[omp_get_max_threads() + 1];
    rowsToFindCnt[0] = 0; colsToFindCnt[0] = 0;

#pragma omp parallel
    {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    int iPerThread = (A->m + nthreads - 1)/nthreads;
    int iBegin = min(iPerThread*tid, A->m);
    int iEnd = min(iBegin + iPerThread, A->m);

    int privateRowCnt = 0;
#pragma omp for
    for (int i = 0; i < A->m; ++i) {
      if (!bv->get(i)) {
        rowsToFindArray[A->m*tid + privateRowCnt] = i;
        ++privateRowCnt;
      }
    }
    rowsToFindCnt[tid + 1] = privateRowCnt;

    int privateColCnt = 0;
#pragma omp for
    for (int j = 0; j < A->n; ++j) {
      if (!bv->get(A->m + j)) {
        colsToFindArray[A->n*tid + privateColCnt] = j;
        ++privateColCnt;
      }
    }
    colsToFindCnt[tid + 1] = privateColCnt;

#pragma omp barrier
#pragma omp master
    {
      for (int i = 1; i < nthreads; ++i) {
        rowsToFindCnt[i + 1] += rowsToFindCnt[i];
        colsToFindCnt[i + 1] += colsToFindCnt[i];
      }
      m = rowsToFindCnt[nthreads];
      n = colsToFindCnt[nthreads];

      rowsToFind = new int[m];
      colsToFind = new int[n];
    }
#pragma omp barrier

    memcpy(
      rowsToFind + rowsToFindCnt[tid], rowsToFindArray + A->m*tid,
      privateRowCnt*sizeof(int));
    memcpy(
      colsToFind + colsToFindCnt[tid], colsToFindArray + A->n*tid,
      privateColCnt*sizeof(int));
    } // omp parallel

    delete[] rowsToFindArray;
    delete[] colsToFindArray;
  } // WITH_BIT_VECTOR

#pragma omp parallel
  {
  int tid = omp_get_thread_num();
  int nthreads = omp_get_num_threads();

  int iPerThread = (m + nthreads - 1)/nthreads;
  int iBegin = min(iPerThread*tid, m);
  int iEnd = min(iBegin + iPerThread, m);

  int jPerTherad = (n + nthreads - 1)/nthreads;
  int jBegin = min(jPerTherad*tid, n);
  int jEnd = min(jBegin + jPerTherad, n);

  for (int i = iBegin; i< iEnd; ++i) {
    int ii = WITH_BIT_VECTOR ? rowsToFind[i] : i;
    p[ii] = ii;
  }
  for (int j = jBegin; j < jEnd; ++j) {
    int jj = (WITH_BIT_VECTOR ? colsToFind[j] : j) + A->m;
    p[jj] = jj;
  }

#pragma omp barrier

  int nnz = A->rowptr[A->m] - BASE;
  int nnzPerThread = (nnz + nthreads - 1)/nthreads;
  int xBegin, xEnd;
  if (WITH_BIT_VECTOR) {
    xBegin = iBegin;
    xEnd = iEnd;
  }
  else {
    xBegin = lower_bound(A->rowptr, A->rowptr + A->m, nnzPerThread*tid + BASE) - A->rowptr;
    xEnd = lower_bound(A->rowptr, A->rowptr + A->m, nnzPerThread*(tid + 1) + BASE) - A->rowptr;
  }
  assert(xBegin <= xEnd);
  assert(xBegin >= 0 && xBegin <= A->m);
  assert(xEnd >= 0 && xEnd <= A->m);

  for (int x = xBegin; x < xEnd; ++x) {
    int xx = WITH_BIT_VECTOR ? rowsToFind[x] : x;
    for (int j = A->rowptr[xx] - BASE; j < A->rowptr[xx + 1] - BASE; ++j) {
      int y = A->colidx[j] + A->m;
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

  if (WITH_BIT_VECTOR) {
    xBegin = jBegin;
    xEnd = jEnd;
  }
  else {
    xBegin = lower_bound(AT->rowptr, AT->rowptr + AT->m, nnzPerThread*tid + BASE) - AT->rowptr;
    xEnd = lower_bound(AT->rowptr, AT->rowptr + AT->m, nnzPerThread*(tid + 1) + BASE) - AT->rowptr;
  }
  assert(xBegin <= xEnd);
  assert(xBegin >= 0 && xBegin <= AT->m);
  assert(xEnd >= 0 && xEnd <= AT->m);

  for (int x = xBegin; x < xEnd; ++x) {
    int xx = WITH_BIT_VECTOR ? colsToFind[x] : x;
    for (int j = AT->rowptr[xx] - BASE; j < AT->rowptr[xx + 1] - BASE; ++j) {
      int y = AT->colidx[j];
      assert(!WITH_BIT_VECTOR || !bv->get(y));
      int xx2 = xx + A->m;
      if (p[xx2] != p[y]) {
        // union
        int r_x = xx2, r_y = y;
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
  } // for each col x

#pragma omp barrier

  // path compression so that all p[i] points to its root
  // and count # of components
  int compId = 0;
  for (int i = iBegin; i < iEnd; ++i) {
    int ii = WITH_BIT_VECTOR ? rowsToFind[i] : i;
    int r = ii;
    while (p[r] != r) {
      r = p[r];
      assert(!WITH_BIT_VECTOR || !bv->get(r));
    }
    p[ii] = r;
    if (r == ii) ++compId;
  }
  for (int j = jBegin; j < jEnd; ++j) {
    int jj = (WITH_BIT_VECTOR ? colsToFind[j] : j) + A->m;
    int r = jj;
    while (p[r] != r) {
      r = p[r];
      assert(!WITH_BIT_VECTOR || !bv->get(r));
    }
    p[jj] = r;
    if (r == jj) ++compId;
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
    int ii = WITH_BIT_VECTOR ? rowsToFind[i] : i;
    int r = p[ii];
    if (r == ii) {
      (*compToRoot)[compId] = r;
      rootToComp[r] = compId;
      ++compId;
    }
  }
  for (int j = jBegin; j < jEnd; ++j) {
    int jj = (WITH_BIT_VECTOR ? colsToFind[j] : j) + A->m;
    int r = p[jj];
    if (r == jj) {
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
    int ii = WITH_BIT_VECTOR ? rowsToFind[i] : i;
    int c = rootToComp[p[ii]];
    ++localPrefixSum[c];
  }
  for (int j = jBegin; j < jEnd; ++j) {
    int jj = (WITH_BIT_VECTOR ? colsToFind[j] : j) + A->m;
    int c = rootToComp[p[jj]];
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

  for (int j = jEnd - 1; j >= jBegin; --j) {
    int jj = (WITH_BIT_VECTOR ? colsToFind[j] : j) + A->m;
    int c = rootToComp[p[jj]];
    --(*compSizePrefixSum)[c + nComp*tid];
    int offset = (*compSizePrefixSum)[c + nComp*tid];
    (*nodesSortedByComp)[offset] = jj;
  }
  
  for (int i = iEnd - 1; i >= iBegin; --i) {
    int ii = WITH_BIT_VECTOR ? rowsToFind[i] : i;
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
  assert(cnt == m + n);

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
  if (rowsToFind) delete[] rowsToFind;
  if (colsToFind) delete[] colsToFind;
}

void findConnectedComponents(
  const CSR *A, const CSR *AT,
  int *numOfComponents, int **compToRoot, int **compSizes, int **compSizePrefixSum,
  int **nodesSortedByComp)
{
  if (0 == A->getBase()) {
    findConnectedComponents_<0>(A, AT, numOfComponents, compToRoot, compSizes, compSizePrefixSum, nodesSortedByComp);
  }
  else {
    assert(1 == A->getBase());
    findConnectedComponents_<1>(A, AT, numOfComponents, compToRoot, compSizes, compSizePrefixSum, nodesSortedByComp);
  }
}

void findConnectedComponentsWithBitVector(
  const CSR *A, const CSR *AT,
  int *numOfComponents, int **compToRoot, int **compSizes, int **compSizePrefixSum,
  int **nodesSortedByComp,
  const BitVector *bv)
{
  if (0 == A->getBase()) {
    findConnectedComponents_<0, true>(A, AT, numOfComponents, compToRoot, compSizes, compSizePrefixSum, nodesSortedByComp, bv);
  }
  else {
    assert(1 == A->getBase());
    findConnectedComponents_<1, true>(A, AT, numOfComponents, compToRoot, compSizes, compSizePrefixSum, nodesSortedByComp, bv);
  }
}
#endif

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

void bfs_serial(
  const CSR *A, const CSR *AT,
  int source,
  BitVector *bv1, BitVector *bv2,
  bfsAuxData *aux,
  int *visited1, int *visited2,
  int *nVisited1, int *nVisited2)
{
  int base = A->getBase();

  int numOfVisited1 = 0, numOfVisited2 = 0;

  int tid = omp_get_thread_num();

  int numLevels = 0;

  int **q = aux->q;
  q[0][tid*max(A->m, A->n)] = source;
  bv1->set(source);

  int qTail = 1;

  while (true) {
    if (qTail == 0) break;

    memcpy(visited1 + numOfVisited1, q[0] + tid*max(A->m, A->n), qTail*sizeof(int));
    numOfVisited1 += qTail;

    int *tailPtr = q[1] + tid*max(A->m, A->n);

    for (int i = 0; i < qTail; ++i) {
      int u = q[0][i + tid*max(A->m, A->n)];

      for (int j = A->rowptr[u] - base; j < A->rowptr[u + 1] - base; ++j) {
        int v = A->colidx[j] - base;
        if (!bv2->get(v)) {
          bv2->set(v);
          *tailPtr = v;
          ++tailPtr;
        }
      }
    }

    qTail = tailPtr - (q[1] + tid*max(A->m, A->n));
    if (qTail == 0) break;

    memcpy(visited2 + numOfVisited2, q[1] + tid*max(A->m, A->n), qTail*sizeof(int));
    numOfVisited2 += qTail;

    tailPtr = q[0] + tid*max(A->m, A->n);

    for (int i = 0; i < qTail; ++i) {
      int u = q[1][i + tid*max(A->m, A->n)];

      for (int j = AT->rowptr[u] - base; j < AT->rowptr[u + 1] - base; ++j) {
        int v = AT->colidx[j] - base;
        if (!bv1->get(v)) {
          bv1->set(v);
          *tailPtr = v;
          ++tailPtr;
        }
      }
    }

    qTail = tailPtr - (q[0] + tid*max(A->m, A->n));
  }

  *nVisited1 = numOfVisited1;
  *nVisited2 = numOfVisited2;

  return;
}

int bfs(
  const CSR *A, const CSR *AT,
  int source,
  BitVector *bv1, BitVector *bv2,
  bfsAuxData *aux,
  int *visited1, int *visited2,
  int *nVisited1, int *nVisited2)
{
  int base = A->getBase();

  *nVisited1 = 0, *nVisited2 = 0;

  int numLevels = 0;
  bv1->set(source);

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

#pragma omp parallel
  {
  int tid = omp_get_thread_num();
  int nthreads = omp_get_num_threads();

  if (tid > 0) {
    qTail[0][tid] = 0;
    nnzPrefixSum[tid + 1] = 0;
  }

  while (true) {
    const CSR *currA, *nextA;
    BitVector *bv;
    int *visited, *nVisited;
    if (numLevels%2 == 0) {
      currA = A;
      nextA = AT;
      bv = bv2;
      visited = visited1;
      nVisited = nVisited1;
    }
    else {
      currA = AT;
      nextA = A;
      bv = bv1;
      visited = visited2;
      nVisited = nVisited2;
    }

    synk::Barrier::getInstance()->wait(tid);
#pragma omp master
    {
      for (int t = 0; t < nthreads; ++t) {
        qTailPrefixSum[t + 1] = qTailPrefixSum[t] + qTail[numLevels%2][t];
        nnzPrefixSum[t + 1] += nnzPrefixSum[t];
      }
      ++numLevels;
      *nVisited += qTailPrefixSum[nthreads];
    }
    synk::Barrier::getInstance()->wait(tid);

    if (qTailPrefixSum[nthreads] == 0) break;

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
          rowptrs + tBegin*max(A->m, A->n), rowptrs + tBegin*max(A->m, A->n) + qTail[1 - numLevels%2][tBegin],
          nnzPerThread*tid - nnzPrefixSum[tBegin]) -
        (rowptrs + tBegin*max(A->m, A->n)) - 1;
    }

    if (tEnd == nthreads) {
      iEnd = 0;
    }
    else {
      iEnd = upper_bound(
          rowptrs + tEnd*max(A->m, A->n), rowptrs + tEnd*max(A->m, A->n) + qTail[1 - numLevels%2][tEnd],
          nnzPerThread*(tid + 1) - nnzPrefixSum[tEnd]) -
        (rowptrs + tEnd*max(A->m, A->n)) - 1;
    }

    memcpy(
      visited + *nVisited - qTailPrefixSum[nthreads] + qTailPrefixSum[tid], 
      q[1 - numLevels%2] + tid*max(A->m, A->n),
      sizeof(int)*(qTailPrefixSum[tid + 1] - qTailPrefixSum[tid]));

    synk::Barrier::getInstance()->wait(tid);

    int *tailPtr = q[numLevels%2] + tid*max(A->m, A->n);
    int *rowptr = rowptrs + tid*max(A->m, A->n);
    *rowptr = 0;

    for (int t = tBegin; t <= tEnd; ++t) {
      for (int i = (t == tBegin ? iBegin : 0);
          i < (t == tEnd ? iEnd : qTail[1 - numLevels%2][t]);
          ++i) {
        int u = q[1 - numLevels%2][t*max(A->m, A->n) + i];

        for (int j = currA->rowptr[u] - base; j < currA->rowptr[u + 1] - base; ++j) {
          int v = currA->colidx[j] - base;
          if (bv->testAndSet(v)) {
            *tailPtr = v;
            *(rowptr + 1) = *rowptr + nextA->rowptr[v + 1] - nextA->rowptr[v];

            ++tailPtr;
            ++rowptr;
          }
        }
      } // for each current node u
    }

    qTail[numLevels%2][tid] = tailPtr - (q[numLevels%2] + tid*max(A->m, A->n));
    nnzPrefixSum[tid + 1] = *rowptr;
  } // while true
  } // omp parallel

#ifndef NDEBUG
  int *temp = new int[*nVisited1];
  copy(visited1, visited1 + *nVisited1, temp);
  sort(temp, temp + *nVisited1);
  for (int i = 0; i < *nVisited1; ++i) {
    assert(temp[i] >= 0 && temp[i] < A->m);
    if (i > 0 && temp[i] == temp[i - 1]) {
      printf("%d duplicated\n", temp[i]);
      assert(false);
    }
  }
  delete[] temp;

  temp = new int[*nVisited2];
  copy(visited2, visited2 + *nVisited2, temp);
  sort(temp, temp + *nVisited2);
  for (int i = 0; i < *nVisited2; ++i) {
    assert(temp[i] >= 0 && temp[i] < AT->m);
    if (i > 0 && temp[i] == temp[i - 1]) {
      printf("%d duplicated\n", temp[i]);
      assert(false);
    }
  }
  delete[] temp;
#endif

  return numLevels;
}

extern "C" {

void bfsBipartite(CSR& A, CSR& AT, int *rowPerm, int *rowInversePerm, int *colPerm, int *colInversePerm)
{
  int base = A.getBase();

  int numOfComponents;
  int *compToRoot, *compSizes, *compSizePrefixSum;
  int *nodesSortedByComp;

  BitVector bv1(A.m);
  BitVector bv2(A.n);

  bfsAuxData aux(max(A.m, A.n));

  int rowPermIdx = 0;
  int colPermIdx = 0;

  int singletonCount = 0;
  int nComponents = 0;

  for (int i = 0; i < A.m; ++i) {
    if (!bv1.get(i)) {
      // short circuit for a singleton or a twin
      if (A.rowptr[i + 1] == A.rowptr[i]) {
        rowInversePerm[rowPermIdx] = i;
        rowPerm[i] = rowPermIdx;
        ++rowPermIdx;
        ++singletonCount;
        ++nComponents;
        continue;
      }

      int nRowVisited, nColVisited;

      int nLevels = bfs(
        &A, &AT, i, &bv1, &bv2, &aux,
        rowInversePerm + rowPermIdx, colInversePerm + colPermIdx,
        &nRowVisited, &nColVisited);
      //printf("nLevels = %d\n", nLevels);

#pragma omp parallel for
      for (int j = 0; j < nRowVisited; ++j) {
        rowPerm[rowInversePerm[rowPermIdx + j]] = rowPermIdx + j;
      }
#pragma omp parallel for
      for (int j = 0; j < nColVisited; ++j) {
        colPerm[colInversePerm[colPermIdx + j]] = colPermIdx + j;
      }

      rowPermIdx += nRowVisited;
      colPermIdx += nColVisited;

      ++nComponents;
    } // for each connected component
  }
  for (int i = 0; i < A.n; ++i) {
    if (!bv2.get(i)) {
      // short circuit for a singleton or a twin
      if (AT.rowptr[i + 1] == AT.rowptr[i]) {
        colInversePerm[colPermIdx] = i;
        colPerm[i] = colPermIdx;
        ++colPermIdx;
        ++singletonCount;
        ++nComponents;
        continue;
      }

      int nRowVisited, nColVisited;

      int nLevels = bfs(
        &AT, &A, i, &bv2, &bv1, &aux,
        colInversePerm + colPermIdx, rowInversePerm + rowPermIdx,
        &nColVisited, &nRowVisited);
      //printf("nLevels = %d\n", nLevels);

#pragma omp parallel for
      for (int j = 0; j < nRowVisited; ++j) {
        rowPerm[rowInversePerm[rowPermIdx + j]] = rowPermIdx + j;
      }
#pragma omp parallel for
      for (int j = 0; j < nColVisited; ++j) {
        colPerm[colInversePerm[colPermIdx + j]] = colPermIdx + j;
      }

      rowPermIdx += nRowVisited;
      colPermIdx += nColVisited;

      ++nComponents;
    } // for each connected component
  }

  //printf("nComponents = %d nSingleton = %d\n", nComponents, singletonCount);

  assert(rowPermIdx == A.m);
  assert(colPermIdx == A.n);

  assert(isPerm(rowPerm, A.m));
  assert(isPerm(rowInversePerm, A.m));
  assert(isPerm(colPerm, A.n));
  assert(isPerm(colInversePerm, A.n));
}

} // extern "C"
