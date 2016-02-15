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

#include <cstdio>
#include <cstring>
#include <climits>
#include <algorithm>

#include <omp.h>

#include "CSR.hpp"
#include "COO.hpp"
#include "mm_io.h"
#include "Utils.hpp"
#include "MemoryPool.hpp"

using namespace std;

namespace SpMP
{

bool CSR::useMemoryPool_() const
{
  return MemoryPool::getSingleton()->contains(rowptr);
}

CSR::CSR() : rowptr(NULL), colidx(NULL), values(NULL), idiag(NULL), diag(NULL), diagptr(NULL), extptr(NULL), ownData_(false)
{
}

void CSR::alloc(int m, int nnz, bool createSeparateDiagData /*= true*/)
{
  this->m = m;

  rowptr = MALLOC(int, m + 1);
  colidx = MALLOC(int, nnz);
  values = MALLOC(double, nnz);
  diagptr = MALLOC(int, m);

  assert(rowptr != NULL);
  assert(colidx != NULL);
  assert(values != NULL);
  assert(diagptr != NULL);

  if (createSeparateDiagData) {
    idiag = MALLOC(double, m);
    diag = MALLOC(double, m);
    assert(idiag != NULL);
    assert(diag != NULL);
  }

  ownData_ = true;
}

CSR::CSR(int m, int n, int nnz)
 : m(m), n(n), extptr(NULL)
{
  alloc(m, nnz);
}

CSR::CSR(const CSR& A) : m(A.m), n(A.n), values(NULL), idiag(NULL), diag(NULL), diagptr(NULL), extptr(NULL), ownData_(true)
{
  int nnz = A.getNnz();

  rowptr = MALLOC(int, m + 1);
  colidx = MALLOC(int, nnz);
  if (A.values) values = MALLOC(double, nnz);
  if (A.diagptr) diagptr = MALLOC(int, m);
  if (A.idiag) idiag = MALLOC(double, m);
  if (A.diag) diag = MALLOC(double, m);

  copyVector(rowptr, A.rowptr, A.m + 1);
  copyVector(colidx, A.colidx, nnz);
  if (values) copyVector(values, A.values, nnz);
  if (diagptr) copyVector(diagptr, A.diagptr, m);
  if (idiag) copyVector(idiag, A.idiag, m);
  if (diag) copyVector(diag, A.diag, m);
}

CSR::CSR(const char *fileName, int base /*=0*/, bool forceSymmetric /*=false*/, int pad /*=1*/)
 : rowptr(NULL), colidx(NULL), values(NULL), ownData_(true), idiag(NULL), diag(NULL), diagptr(NULL), extptr(NULL)
{
  int m = atoi(fileName);
  char buf[1024];
  sprintf(buf, "%d", m);

  int l = strlen(fileName);

  if (!strcmp(buf, fileName)) {
    generate3D27PtLaplacian(this, m, base);
  }
  else if (l > 4 && !strcmp(fileName + l - 4, ".bin")) {
    loadBin(fileName, base);
  }
  else {
    COO Acoo;
    loadMatrixMarket((char *)fileName, Acoo, forceSymmetric, pad);

    alloc(Acoo.m, Acoo.nnz);

    dcoo2csr(this, &Acoo, base);
  }
}

CSR::CSR(int m, int n, int *rowptr, int *colidx, double *values) :
 m(m), n(n), rowptr(rowptr), colidx(colidx), values(values), ownData_(false), idiag(NULL), diag(NULL), extptr(NULL), diagptr(NULL)
{
  assert(getBase() == 0 || getBase() == 1);
}

void CSR::dealloc()
{
  if (useMemoryPool_()) {
    // a large single contiguous chunk is allocated to
    // buffers except rowptr and colidx.
    rowptr = NULL;
    extptr = NULL;
    colidx = NULL;
    values = NULL;
    idiag = NULL;
    diag = NULL;
    diagptr = NULL;
  }
  else {
    if (ownData_) {
      FREE(rowptr);
      FREE(extptr);
      FREE(colidx);
      FREE(values);
    }

    FREE(idiag);
    FREE(diag);
    FREE(diagptr);
  }
}

CSR::~CSR()
{
  dealloc();
}

bool CSR::isSymmetric(bool checkValues, bool printFirstNonSymmetry) const
{
  if (m != n && !extptr) {
    if (printFirstNonSymmetry) {
      printf("Not square\n");
    }
    return false;
  }

  int base = getBase();

  const int *rowptr = this->rowptr - base;
  const int *extptr = this->extptr ? this->extptr - base : rowptr + 1;
  const int *colidx = this->colidx - base;
  const double *values = this->values ? this->values - base : NULL;

  if (!printFirstNonSymmetry) {
    volatile bool isSymmetric = true;

#pragma omp parallel
    {
      int begin, end;
      getSimpleThreadPartition(&begin, &end, m);
      begin += base;
      end += base;

      for (int i = begin; i < end; ++i) {
        for (int j = rowptr[i]; j < extptr[i]; ++j) {
          int c = colidx[j];
          if (c != i) {
            bool hasPair = false;
            for (int k = rowptr[c]; k < extptr[c]; ++k) {
              if (colidx[k] == i) {
                hasPair = true;
                if (checkValues && values[j] != values[k]) {
                  isSymmetric = false;
                }
                break;
              }
            }
            if (!hasPair) {
              isSymmetric = false;
            }
          }
        }
        if (!isSymmetric) break;
      } // for each row
    } // omp parallel

    return isSymmetric;
  }
  else {
    for (int i = base; i < m + base; ++i) {
      for (int j = rowptr[i]; j < extptr[i]; ++j) {
        int c = colidx[j];
        if (c != i) {
          bool hasPair = false;
          for (int k = rowptr[c]; k < extptr[c]; ++k) {
            if (colidx[k] == i) {
              hasPair = true;
              if (checkValues && values[j] != values[k]) {
                printf(
                  "assymmetric (%d, %d) = %g, (%d, %d) = %g\n", 
                  i + 1, c + 1, values[j], c + 1, i + 1, values[k]);
                return false;
              }
              break;
            }
          }
          if (!hasPair) {
            printf(
              "assymmetric (%d, %d) exists but (%d, %d) doesn't\n",
              i + 1, c + 1, c + 1, i + 1);
            return false;
          }
        }
      }
    } // for each row
  }

  return true;
}

void CSR::storeMatrixMarket(const char *fileName) const
{
  int base = getBase();

  FILE *fp = fopen(fileName, "w");
  assert(fp);

  MM_typecode matcode;
  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_sparse(&matcode);
  mm_set_real(&matcode);

  // print banner followed by typecode.
  fprintf(fp, "%s ", MatrixMarketBanner);
  fprintf(fp, "%s\n", mm_typecode_to_str(matcode));

  // print matrix size and nonzeros.
  fprintf(fp, "%d %d %d\n", m, n, rowptr[m] - base);
  
  // print values
  for (int i = 0; i < m; ++i) {
    for (int j = rowptr[i] - base; j < rowptr[i + 1] - base; ++j) {
      fprintf(fp, "%d %d %20.16g\n", i + 1, colidx[j] - base + 1, values[j]);
    }
  }

  fclose(fp);
}

static const int MAT_FILE_CLASSID = 1211216;

void CSR::loadBin(const char *file_name, int base /*=0*/)
{
  dealloc();
 
  FILE *fp = fopen(file_name, "r");
  if (!fp) {
    fprintf(stderr, "Failed to open %s\n", file_name);
    return;
  }
 
  int id;
  fread(&id, sizeof(id), 1, fp);
  if (MAT_FILE_CLASSID != id) {
    fprintf(stderr, "Wrong file ID (%d)\n", id);
  }
 
  fread(&m, sizeof(m), 1, fp);
  fread(&n, sizeof(n), 1, fp);
  int nnz;
  fread(&nnz, sizeof(nnz), 1, fp);
 
  alloc(m, nnz);
 
  fread(rowptr + 1, sizeof(rowptr[0]), m, fp);
  rowptr[0] = 0;
  for (int i = 1; i < m; ++i) {
    rowptr[i + 1] += rowptr[i];
  }
 
  fread(colidx, sizeof(colidx[0]), nnz, fp);
  fread(values, sizeof(values[0]), nnz, fp);
 
#pragma omp parallel for
  for (int i = 0; i < m; ++i) {
    for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
      if (colidx[j] == i) {
        diagptr[i] = j;
        diag[i] = values[j];
        idiag[i] = 1/values[j];
      }
    }
  }
 
  fclose(fp);

  if (1 == base) {
    make1BasedIndexing();
  }
  else {
    assert(0 == base);
  }
}

void CSR::storeBin(const char *fileName)
{
  bool was1Based = 1 == getBase();
  if (was1Based) {
    make0BasedIndexing();
  }

  FILE *fp = fopen(fileName, "w");
  if (!fp) {
    fprintf(stderr, "Failed to open %s\n", fileName);
    return;
  }

  int id = MAT_FILE_CLASSID;
  fwrite(&id, sizeof(id), 1, fp);

  fwrite(&m, sizeof(m), 1, fp);
  fwrite(&n, sizeof(n), 1, fp);
  int nnz = getNnz();
  fwrite(&nnz, sizeof(nnz), 1, fp);

  int *rownnz = (int *)malloc(sizeof(int)*m);
  for (int i = 0; i < m; ++i) {
    rownnz[i] = rowptr[i + 1] - rowptr[i];
  }
  fwrite(rownnz, sizeof(rownnz[0]), m, fp);
  free(rownnz);

  fwrite(colidx, sizeof(colidx[0]), nnz, fp);
  fwrite(values, sizeof(values[0]), nnz, fp);

  fclose(fp);

  if (was1Based) {
    make1BasedIndexing();
  }
}

void CSR::make0BasedIndexing()
{
  if (0 == getBase()) return;

  int nnz = getNnz();

#pragma omp parallel for
  for(int i=0; i <= m; i++)
    rowptr[i]--;

#pragma omp parallel for
  for(int i=0; i < nnz; i++)
    colidx[i]--;

  if (diagptr) {
#pragma omp parallel for
    for (int i = 0; i < m; i++)
      diagptr[i]--;
  }
}

void CSR::make1BasedIndexing()
{
  if (1 == getBase()) return;

  int nnz = getNnz();

#pragma omp parallel for
  for(int i=0; i <= m; i++)
    rowptr[i]++;

#pragma omp parallel for
  for(int i=0; i < nnz; i++)
    colidx[i]++;

  if (diagptr) {
#pragma omp parallel for
    for (int i = 0; i < m; i++)
      diagptr[i]++;
  }
}

void CSR::computeInverseDiag()
{
  if (!idiag) {
    constructDiagPtr();

    int base = getBase();
    const double *values = this->values - base;

    idiag = MALLOC(double, m);
#pragma omp parallel for
    for (int i = 0; i < m; i++) {
      idiag[i] = 1/values[diagptr[i]];
    }
  }
}

/**
 * idx = idx2*dim1 + idx1
 * -> ret = idx1*dim2 + idx2
 *        = (idx%dim1)*dim2 + idx/dim1
 */
static inline int transpose_idx(int idx, int dim1, int dim2)
{
  return idx%dim1*dim2 + idx/dim1;
}

// TODO: only supports local matrix for now
/**
 * Transposition using parallel counting sort
 */
CSR *CSR::transpose() const
{
  int base = getBase();
  const int *rowptr = this->rowptr - base;
  const int *colidx = this->colidx - base;
  const double *values = this->values ? this->values - base : NULL;

  int nnz = getNnz();

  CSR *AT = new CSR();

  AT->m = n;
  AT->n = m;
  AT->colidx = MALLOC(int, nnz);
  assert(AT->colidx);
  if (values) {
    AT->values = MALLOC(double, nnz);
    assert(AT->values);
  }
  if (diagptr) {
    AT->diagptr = MALLOC(int, n);
    assert(AT->diagptr);
  }

  if (0 == n) {
    return AT;
  }

  double t = omp_get_wtime();

  int *bucket = MALLOC(int, (n + 1)*omp_get_max_threads());
  bucket -= base;

#ifndef NDEBUG
  int i;
  for (i = base; i < m + base; ++i) {
    assert(rowptr[i + 1] >= rowptr[i]);
  }
#endif

#pragma omp parallel
  {
  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();

  int iBegin, iEnd;
  getLoadBalancedPartition(&iBegin, &iEnd, rowptr + base, m);
  iBegin += base;
  iEnd += base;

  int i, j;
  memset(bucket + base + tid*n, 0, sizeof(int)*n);

  // count the number of keys that will go into each bucket
  for (j = rowptr[iBegin]; j < rowptr[iEnd]; ++j) {
    int idx = colidx[j];
#ifndef NDEBUG
    if (idx < base || idx >= n + base) {
      printf("tid = %d m = %d n = %d nnz = %d iBegin = %d iEnd = %d rowptr[iBegin] = %d rowptr[iEnd] = %d j = %d idx = %d\n", tid, m, n, nnz, iBegin, iEnd, rowptr[iBegin], rowptr[iEnd], j, idx);
    }
#endif
    assert(idx >= base && idx < n + base);
    bucket[tid*n + idx]++;
  }
  // up to here, bucket is used as int[nthreads][n] 2D array

  // prefix sum
#pragma omp barrier

  for (i = tid*n + 1; i < (tid + 1)*n; ++i) {
    int transpose_i = transpose_idx(i, nthreads, n);
    int transpose_i_minus_1 = transpose_idx(i - 1, nthreads, n);

    bucket[transpose_i + base] += bucket[transpose_i_minus_1 + base];
  }

#pragma omp barrier
#pragma omp master
  {
    for (i = 1; i < nthreads; ++i) {
      int j0 = n*i - 1, j1 = n*(i + 1) - 1;
      int transpose_j0 = transpose_idx(j0, nthreads, n);
      int transpose_j1 = transpose_idx(j1, nthreads, n);

      bucket[transpose_j1 + base] += bucket[transpose_j0 + base];
    }
  }
#pragma omp barrier

  if (tid > 0) {
    int transpose_i0 = transpose_idx(n*tid - 1, nthreads, n);

    for (i = tid*n; i < (tid + 1)*n - 1; ++i) {
      int transpose_i = transpose_idx(i, nthreads, n);

      bucket[transpose_i + base] += bucket[transpose_i0 + base];
    }
  }

#pragma omp barrier

  if (values) {
    for (i = iEnd - 1; i >= iBegin; --i) {
      for (j = rowptr[i + 1] - 1; j >= rowptr[i]; --j) {
        int idx = colidx[j];
        --bucket[tid*n + idx];

        int offset = bucket[tid*n + idx];

        assert(offset >= 0 && offset < nnz);
        AT->values[offset] = values[j];
        AT->colidx[offset] = i;
      }
    }
  }
  else {
    for (i = iEnd - 1; i >= iBegin; --i) {
      for (j = rowptr[i + 1] - 1; j >= rowptr[i]; --j) {
        int idx = colidx[j];
        --bucket[tid*n + idx];

        int offset = bucket[tid*n + idx];

        AT->colidx[offset] = i;
      }
    }
  }

  if (base > 0) {
#pragma omp barrier
#pragma omp for
    for (int i = base; i < n + base; i++) {
      bucket[i] += base;
    }
    assert(bucket[base] == base);
  }

  if (diagptr) {
#pragma omp barrier

    bucket[n + base] = nnz + base;

    int *AT_colidx = AT->colidx - base;
    int *AT_diagptr = AT->diagptr - base;

#pragma omp for
    for (int i = base; i < n + base; i++) {
      for (int j = bucket[i]; j < bucket[i + 1]; ++j) {
        int c = AT_colidx[j];
        if (c == i) AT_diagptr[i] = j;
      }
    }
  }

  } // omp parallel

  bucket[n + base] = nnz + base;
  AT->rowptr = bucket + base; 

  return AT;
}

int CSR::getBandwidth() const
{
  int base = getBase();
  const int *rowptr = this->rowptr - base;
  const int *colidx = this->colidx - base;

  int bw = INT_MIN;
#pragma omp parallel reduction(max:bw)
  {
    int iBegin, iEnd;
    getLoadBalancedPartition(&iBegin, &iEnd, rowptr + base, m);
    iBegin += base;
    iEnd += base;

    for (int i = iBegin; i < iEnd; ++i) {
      for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
        int c = colidx[j];
        int temp = c - i;
        if (temp < 0) temp = -temp;
        bw = max(temp, bw);
      }
    }
  } // omp parallel

  return bw;
}

double CSR::getAverageWidth(bool sorted /*= false*/) const
{
  int base = getBase();
  const int *rowptr = this->rowptr - base;
  const int *colidx = this->colidx - base;

  unsigned long long total_width = 0;
  if (sorted) {
#pragma omp parallel for reduction(+:total_width)
    for (int i = base; i < m + base; ++i) {
      if (rowptr[i] == rowptr[i + 1]) continue;

      int width = colidx[rowptr[i + 1] - 1] - colidx[rowptr[i]];
      total_width += width;
    }
  }
  else {
#pragma omp parallel reduction(+:total_width)
    {
      int iBegin, iEnd;
      getLoadBalancedPartition(&iBegin, &iEnd, rowptr + base, m);
      iBegin += base;
      iEnd += base;

      for (int i = iBegin; i < iEnd; ++i) {
        if (rowptr[i] == rowptr[i + 1]) continue;

        int min_row = INT_MAX, max_row = INT_MIN;
        for (int j = rowptr[i]; j < rowptr[i + 1]; ++j) {
          min_row = min(colidx[j], min_row);
          max_row = max(colidx[j], max_row);
        }

        int width = max_row - min_row;
        total_width += width;
      }
    } // omp parallel
  }

  return (double)total_width/m;
}

bool CSR::equals(const CSR& A, bool print /*=false*/) const
{
  int base = getBase();
  if (A.getBase() != base) {
    if (print) printf("base differs %d vs. %d\n", base, A.getBase());
    return false;
  }
  if (m != A.m) {
    if (print) printf("number of rows differs %d vs. %d\n", m, A.m);
    return false;
  }
  if (n != A.n) {
    if (print) printf("number of columns differs %d vs. %d\n", n, A.n);
    return false;
  }
  if (rowptr[m] != A.rowptr[A.m]) {
    if (print) printf("number of non-zeros differs %d vs. %d\n", rowptr[m], A.rowptr[A.m]);
    return false;
  }
  for (int i = 0; i < m; ++i) {
    if (rowptr[i] != A.rowptr[i]) {
      if (print) printf("rowptr[%d] differs %d vs. %d\n", i, rowptr[i], A.rowptr[i]);
      return false;
    }
    for (int j = rowptr[i] - base; j < rowptr[i + 1] - base; ++j) {
      if (colidx[j] != A.colidx[j]) {
        if (print) printf("colidx[%d:%d] differs %d vs. %d\n", i, j, colidx[j], A.colidx[j]);
        return false;
      }
      if (values[j] != A.values[j]) {
        if (print) printf("values[%d:%d] differs %g vs. %g\n", i, j, values[j], A.values[j]);
        return false;
      }
    }
  } // for each row

  return true;
}

void CSR::print() const
{
  int base = getBase();
  for (int i = 0; i < m; i++) {
    for (int j = rowptr[i] - base; j < rowptr[i + 1] - base; j++) {
      printf("%d %d %g\n", i + base, colidx[j], values[j]);
    }
  }
}

} // namespace SpMP
