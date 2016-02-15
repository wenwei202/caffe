#include <algorithm>
#include <cstring>

#include "CSR.hpp"

using namespace std;
using namespace SpMP;

namespace SpMP
{

void splitLU(CSR& A, CSR *L, CSR *U)
{
  int oldBase = A.getBase();
  A.make0BasedIndexing();

  L->dealloc();
  U->dealloc();

  L->m = U->m = A.m;
  L->n = U->n = A.n;

  const int *extptr = A.extptr ? A.extptr : A.rowptr + 1;
  if (A.extptr) {
    U->extptr = MALLOC(int, U->m);
  }

  L->rowptr = MALLOC(int, L->m + 1);
  U->rowptr = MALLOC(int, U->m + 1);
  L->idiag = MALLOC(double, L->m);
  U->idiag = MALLOC(double, U->m);

  // Count # of nnz per row
  int rowPtrPartialSum[2][omp_get_max_threads() + 1];
  rowPtrPartialSum[0][0] = rowPtrPartialSum[1][0] = 0;

#pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    int iBegin, iEnd;
    getSimpleThreadPartition(&iBegin, &iEnd, A.m);

    // count # of nnz per row
    int sum0 = 0, sum1 = 0;
    for (int i = iBegin; i < iEnd; ++i) {
      L->rowptr[i] = sum0;
      U->rowptr[i] = sum1;

      for (int j = A.rowptr[i]; j < extptr[i]; ++j) {
        if (A.colidx[j] < i) sum0++;
        if (A.colidx[j] > i) sum1++;
      } // for each element
      sum1 += A.rowptr[i + 1] - extptr[i];
    } // for each row

    rowPtrPartialSum[0][tid + 1] = sum0;
    rowPtrPartialSum[1][tid + 1] = sum1;

#pragma omp barrier
#pragma omp single
    {
      for (int i = 1; i < nthreads; ++i) {
        rowPtrPartialSum[0][i + 1] += rowPtrPartialSum[0][i];
        rowPtrPartialSum[1][i + 1] += rowPtrPartialSum[1][i];
      }
      L->rowptr[L->m] = rowPtrPartialSum[0][nthreads];
      U->rowptr[U->m] = rowPtrPartialSum[1][nthreads];

      int nnzL = L->rowptr[L->m];
      int nnzU = U->rowptr[U->m];

      L->colidx = MALLOC(int, nnzL);
      L->values = MALLOC(double, nnzL);

      U->colidx = MALLOC(int, nnzU);
      U->values = MALLOC(double, nnzU);
    }

    for (int i = iBegin; i < iEnd; ++i) {
      L->rowptr[i] += rowPtrPartialSum[0][tid];
      U->rowptr[i] += rowPtrPartialSum[1][tid];

      if (A.extptr && i > iBegin) {
        U->extptr[i - 1] = U->rowptr[i] - (A.rowptr[i] - extptr[i - 1]);
      }

      int idx0 = L->rowptr[i], idx1 = U->rowptr[i];
      for (int j = A.rowptr[i]; j < A.rowptr[i + 1]; ++j) {
        if (A.colidx[j] < i) {
          L->colidx[idx0] = A.colidx[j];
          L->values[idx0] = A.values[j];
          ++idx0;
        }
        if (A.colidx[j] > i) {
          U->colidx[idx1] = A.colidx[j];
          U->values[idx1] = A.values[j];
          ++idx1;
        }
      }

      L->idiag[i] = A.idiag[i];
      U->idiag[i] = A.idiag[i];
    } // for each row

    if (A.extptr && iEnd > iBegin) {
      U->extptr[iEnd - 1] =
        rowPtrPartialSum[1][tid + 1] - (A.rowptr[iEnd] - extptr[iEnd - 1]);
    }
  } // omp parallel

  if (1 == oldBase) {
    A.make1BasedIndexing();
    L->make1BasedIndexing();
    U->make1BasedIndexing();
  }
}

bool getSymmetricNnzPattern(
  const CSR *A, int **symRowPtr, int **symDiagPtr, int **symExtPtr, int **symColIdx)
{
  int base = A->getBase();

  int m = A->m;
  const int *rowptr = A->rowptr - base;
  const int *colidx = A->colidx - base;
  const int *extptr = A->extptr ? A->extptr - base : rowptr + 1;

  size_t symRowPtrBegin;
  if (A->useMemoryPool_()) {
    symRowPtrBegin = MemoryPool::getSingleton()->getTail();
  }
  *symRowPtr = A->allocate_<int>(m + 1);
  (*symRowPtr)[0] = base;
  *symRowPtr -= base;
  int *cnts = NULL;

  volatile bool isSymmetric = true;

//#define PRINT_TIME_BREAKDOWN

#pragma omp parallel
  {
    int tid = omp_get_thread_num();

    int iBegin, iEnd;
    getSimpleThreadPartition(&iBegin, &iEnd, m);
    iBegin += base;
    iEnd += base;

#ifdef PRINT_TIME_BREAKDOWN
    unsigned long long t = __rdtsc();
#endif

    // construct symRowPtr
    for (int i = iBegin; i < iEnd; ++i) {
      (*symRowPtr)[i + 1] = rowptr[i + 1] - rowptr[i];
    }
#pragma omp barrier

#ifdef PRINT_TIME_BREAKDOWN
    if (0 == tid) {
      printf("counting fwd dependencies takes %f\n", (__rdtsc() - t)/get_cpu_freq());
    }
    t = __rdtsc();
#endif

    volatile bool localIsSymmetric = true;
    for (int i = iBegin; i < iEnd; ++i) {
      for (int j = rowptr[i]; j < extptr[i]; ++j) {
        int c = colidx[j];
        // assume colidx is sorted
        if (!binary_search(colidx + rowptr[c], colidx + extptr[c], i)) {
          // for each (i, c), add (c, i)
          __sync_fetch_and_add(*symRowPtr + c + 1, 1);
          localIsSymmetric = false;
        }
      }
    }

    if (!localIsSymmetric) isSymmetric = false;

#pragma omp barrier
#ifdef PRINT_TIME_BREAKDOWN
    if (0 == tid) {
      printf("counting bwd dependencies takes %f\n", (__rdtsc() - t)/get_cpu_freq());
    }
    t = __rdtsc();
#endif
    if (!isSymmetric) {
#pragma omp single
      {
        // FIXME - parallel prefix sum
        for (int i = base; i < m + base; ++i) {
          (*symRowPtr)[i + 1] += (*symRowPtr)[i];
        }
        *symColIdx = MALLOC(int, (*symRowPtr)[m]);
        *symColIdx -= base;

        *symDiagPtr = MALLOC(int, m + 1);
        *symDiagPtr -= base;
        if (A->extptr) {
          *symExtPtr = MALLOC(int, m + 1);
          *symExtPtr -= base;
        }
        cnts = MALLOC(int, m);
      }

#ifdef PRINT_TIME_BREAKDOWN
      if (0 == tid) {
        printf("prefix sum of symRowPtr takes %f\n", (__rdtsc() - t)/get_cpu_freq());
      }
      t = __rdtsc();
#endif

      // construct symColIdx

      // forward direction
      for (int i = iBegin; i < iEnd; ++i) {
        cnts[i] = extptr[i] - rowptr[i];
        memcpy(
          *symColIdx + (*symRowPtr)[i], colidx + rowptr[i],
          cnts[i]*sizeof(int));
      }
#pragma omp barrier
#ifdef PRINT_TIME_BREAKDOWN
      if (0 == tid) {
        printf("construct forward symColIdx takes %f\n", (__rdtsc() - t)/get_cpu_freq());
      }
      t = __rdtsc();
#endif

      // backward direction
      for (int i = iBegin; i < iEnd; ++i) {
        for (int j = rowptr[i]; j < extptr[i]; ++j) {
          int c = colidx[j];
          if (!binary_search(colidx + rowptr[c], colidx + extptr[c], i)) {
            // for each (i, c), add (c, i)
            int cnt = __sync_fetch_and_add(cnts + c, 1);
            (*symColIdx)[(*symRowPtr)[c] + cnt] = i;
          }
        }
      }
#pragma omp barrier
#ifdef PRINT_TIME_BREAKDOWN
      if (0 == tid) {
        printf("construct backward symColIdx takes %f\n", (__rdtsc() - t)/get_cpu_freq());
      }
      t = __rdtsc();
#endif

      // sort colidx and copy remote
      for (int i = iBegin; i < iEnd; ++i) {
        sort(*symColIdx + (*symRowPtr)[i], *symColIdx + (*symRowPtr)[i] + cnts[i]);
        memcpy(
          *symColIdx + (*symRowPtr)[i] + cnts[i], colidx + extptr[i],
          (rowptr[i + 1] - extptr[i])*sizeof(int));

        if (A->extptr) (*symExtPtr)[i] = (*symRowPtr)[i] + cnts[i];

        for (int j = (*symRowPtr)[i]; j < (*symRowPtr)[i + 1]; ++j) {
          if ((*symColIdx)[j] == i) (*symDiagPtr)[i] = j;
        }
      }
#ifdef PRINT_TIME_BREAKDOWN
      if (0 == tid) {
        printf("sorting symColIdx takes %f\n", (__rdtsc() - t)/get_cpu_freq());
      }
#undef PRINT_TIME_BREAKDOWN
#endif

      *symColIdx += base;
      if (*symExtPtr) *symExtPtr += base;
    } // !isSymmetric
  } // omp parallel

  *symRowPtr += base;

  if (isSymmetric) {
    if (!A->useMemoryPool_()) {
      FREE(*symRowPtr);
    }
    else {
      MemoryPool::getSingleton()->setTail(symRowPtrBegin);
      *symRowPtr = NULL;
    }
  }

  FREE(cnts);

#ifndef NDEBUG
  if (isSymmetric) {
    assert(A->isSymmetric(false, true));
  }
  else {
    CSR sym;
    sym.m = m;
    sym.n = m;
    sym.rowptr = *symRowPtr;
    sym.colidx = *symColIdx;
    sym.diagptr = *symDiagPtr;
    sym.extptr = *symExtPtr;

    assert(sym.isSymmetric(false, true));

    sym.rowptr = NULL;
    sym.colidx = NULL;
    sym.diagptr = NULL;
    sym.extptr = NULL;
  }
#endif

  return isSymmetric;
}

} // namespace SpMP
