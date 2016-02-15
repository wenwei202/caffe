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
#include <algorithm>

#include "COO.hpp"
#include "mm_io.h"

using namespace std;

namespace SpMP
{

COO::COO() : rowidx(NULL), colidx(NULL), values(NULL), isSymmetric(false)
{
}

COO::~COO()
{
  dealloc();
}

void COO::dealloc()
{
  FREE(rowidx);
  FREE(colidx);
  FREE(values);
}

void COO::storeMatrixMarket(const char *fileName) const
{
  FILE *fp = fopen(fileName, "w");
  assert(fp);

  MM_typecode matcode;
  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_sparse(&matcode);
  mm_set_real(&matcode);

  int err = mm_write_mtx_crd(
    (char *)fileName, m, n, nnz, rowidx, colidx, values, matcode);
  if (err) {
    fprintf(
      stderr,
      "Fail to write matrix to %s (error code = %d)\n", fileName, err);
    exit(-1);
  }
}

template<class T>
static void qsort(int *idx, T *w, int left, int right)
{
  if (left >= right) return;

  swap(idx[left], idx[left + (right - left)/2]);
  swap(w[left], w[left + (right - left)/2]);

  int last = left;
  for (int i = left+1; i <= right; i++) {
    if (idx[i] < idx[left]) {
       ++last;
       swap(idx[last], idx[i]);
       swap(w[last], w[i]);
    }
  }

  swap(idx[left], idx[last]);
  swap(w[left], w[last]);

  qsort(idx, w, left, last-1);
  qsort(idx, w, last+1, right);
}

/* converts COO format to CSR format, not in-place,
   if SORT_IN_ROW is defined, each row is sorted in column index.
assume COO is one-based index */

template<class T>
void coo2csr(
  int m, int nnz,
  int *rowptr, int *colidx, T *values,
  const int *cooRowidx, const int *cooColidx, const T *cooValues,
  bool sort,
  int outBase = 0)
{
  int i, l;

#pragma omp parallel for
  for (i = 0; i <= m; i++) rowptr[i] = 0;

  /* determine row lengths */
  for (i = 0; i < nnz; i++) rowptr[cooRowidx[i]]++;

  for (i = 0; i < m; i++) rowptr[i+1] += rowptr[i];

  /* go through the structure  once more. Fill in output matrix. */
  for (l = 0; l < nnz; l++) {
    i = rowptr[cooRowidx[l] - 1];
    values[i] = cooValues[l];
    colidx[i] = cooColidx[l] - 1 + outBase;
    rowptr[cooRowidx[l] - 1]++;
  }

  /* shift back rowptr */
  for (i = m; i > 0; i--) rowptr[i] = rowptr[i-1] + outBase;

  rowptr[0] = outBase;

  if (sort) {
#pragma omp parallel for
    for (i=0; i < m; i++){
      qsort(colidx, values, rowptr[i] - outBase, rowptr[i+1] - 1 - outBase);
      assert(is_sorted(colidx + rowptr[i] - outBase, colidx + rowptr[i+1] - outBase));
    }
  }
}

void dcoo2csr(
  int m, int nnz,
  int *rowptr, int *colidx, double *values,
  const int *cooRowidx, const int *cooColidx, const double *cooValues,
  bool sort /*=true*/,
  int outBase /*=0*/)
{
  coo2csr(m, nnz, rowptr, colidx, values, cooRowidx, cooColidx, cooValues, sort, outBase);
}

void dcoo2csr(CSR *Acrs, const COO *Acoo, int outBase /*=0*/, bool createSeparateDiagData /*= true*/)
{
  Acrs->n=Acoo->n;
  Acrs->m=Acoo->m;

  dcoo2csr(
    Acrs->m, Acoo->nnz,
    Acrs->rowptr, Acrs->colidx, Acrs->values,
    Acoo->rowidx, Acoo->colidx, Acoo->values,
    true /*sort*/, outBase);

  int base = Acrs->getBase();
  if (Acrs->diagptr) {
    if (!Acrs->idiag || !Acrs->diag) {
      createSeparateDiagData = false;
    }
#pragma omp parallel for
    for (int i = 0; i < Acrs->m; ++i) {
      for (int j = Acrs->rowptr[i] - base; j < Acrs->rowptr[i + 1] - base; ++j) {
        if (Acrs->colidx[j] - base == i) {
          Acrs->diagptr[i] = j + base;

          if (createSeparateDiagData) {
            Acrs->idiag[i] = 1/Acrs->values[j];
            Acrs->diag[i] = Acrs->values[j];
          }
        }
      }
    }
  }
}

static bool loadMatrixMarket_(const char *file, COO &coo, bool force_symmetric, bool transpose, int pad)
{
  FILE *fp=fopen(file, "r");
  if (NULL == fp) {
    fprintf(stderr, "Failed to open file %s\n", file);
    exit(-1);
  }

  // read banner
  MM_typecode matcode;
  if (mm_read_banner (fp, &matcode) != 0) {
    fprintf(stderr, "Error: could not process Matrix Market banner.\n");
    fclose(fp);
    return false;
  }

  if (!mm_is_valid (matcode) || mm_is_array (matcode) || mm_is_dense (matcode) ) {
    fprintf(stderr, "Error: only support sparse and real matrices.\n");
    fclose(fp);
    return false;
  }
  bool pattern = mm_is_pattern (matcode);

  // read sizes
  int m, n;
  int nnz; // # of non-zeros specified in the file
  if (mm_read_mtx_crd_size(fp, &m, &n, &nnz) !=0) {
    fprintf(stderr, "Error: could not read matrix size.\n");
    fclose(fp);
    return false;
  }
  if (transpose) {
    assert(!force_symmetric);
    swap(m, n);
  }

  int origM = m, origN = n;
  m = (m + pad - 1)/pad*pad;
  n = (n + pad - 1)/pad*pad;

  size_t count;
  if (force_symmetric || mm_is_symmetric (matcode) == 1) {
    coo.isSymmetric = true;
    count = 2L*nnz;
  }
  else {
    count = nnz;
  }

  // allocate memory
  size_t extraCount = min(m, n) - min(origM, origN);
  double *values = MALLOC(double, count + extraCount);
  int *colidx = MALLOC(int, count + extraCount);
  int *rowidx = MALLOC(int, count + extraCount);
  if (!values || !colidx || !rowidx) {
    fprintf(stderr, "Failed to allocate memory\n");
    fclose(fp);
    return false;
  }

  int *colidx_temp, *rowcnt = NULL;
  if (coo.isSymmetric) {
    colidx_temp = MALLOC(int, count);
    rowcnt = MALLOC(int, m + 1);
    if (!colidx_temp || !rowcnt) {
      fprintf(stderr, "Failed to allocate memory\n");
      fclose(fp);
      return false;
    }
    memset(rowcnt, 0, sizeof(int)*(m + 1));
  }

  // read values
  count = 0;
  int lines = 0;
  int x, y;
  double real, imag;
  int base = 1;
  while (mm_read_mtx_crd_entry (fp, &x, &y, &real, &imag, matcode) == 0) {
    if (transpose) swap(x, y);

    if (x > origM || y > origN) {
      fprintf(stderr, "Error: (%d %d) coordinate is out of range.\n", x, y);
      fclose(fp);
      return false;
    }

    rowidx[count] = x;
    colidx[count] = y;
    values[count] = pattern ? 1 : real;
    if (0 == x || 0 == y) base = 0;

    ++count;
    ++lines;
    if (coo.isSymmetric) rowcnt[x]++;
      // this is not a bug. we're intentionally indexing rowcnt[x] instead of rowcnt[x-1]
  }
  // padding for vectorization
  for (int i = min(origM, origN); i < min(m, n); ++i) {
    rowidx[count] = i + 1;
    colidx[count] = i + 1;
    values[count] = 1;
    ++count;
  }
  fclose(fp);

  if (0 == base) {
    for (int i = 0; i < count; ++i) {
      rowidx[i]++;
      colidx[i]++;
    }
    if (coo.isSymmetric) {
      for (int i = m; i > 0; --i) {
        rowcnt[i] = rowcnt[i - 1];
      }
    } 
  }

  if (lines != nnz) {
    fprintf(stderr, "Error: nnz (%d) specified in the header doesn't match with # of lines (%d) in file %s\n",
            nnz, lines, file);
    return false;
  }
    
  if (coo.isSymmetric) {
    // add transposed elements only if it doesn't exist
    size_t real_count = count;
    // preix-sum
    for (int i = 0; i < m; ++i) {
      rowcnt[i + 1] += rowcnt[i];
    }
    for (int i = 0; i < count; ++i) {
      int j = rowcnt[rowidx[i] - 1];
      colidx_temp[j] = colidx[i];
      rowcnt[rowidx[i] - 1]++;
    }
    for (int i = m; i > 0; --i) {
      rowcnt[i] = rowcnt[i - 1];
    }
    rowcnt[0] = 0;

#pragma omp parallel for
    for (int i = 0; i < m; ++i) {
      sort(colidx_temp + rowcnt[i], colidx_temp + rowcnt[i + 1]);
    }

    for (int i = 0; i < count; ++i) {
      int x = rowidx[i], y = colidx[i];
      if (x != y) {
        if (!binary_search(
            colidx_temp + rowcnt[y - 1], colidx_temp + rowcnt[y], x)) {
          rowidx[real_count] = y;
          colidx[real_count] = x;
          values[real_count] = values[i];
          ++real_count;
        }
      }
    }
    count = real_count;

    FREE(rowcnt);
    FREE(colidx_temp);
  }

  coo.m = m;
  coo.n = n;
  coo.nnz = count;
  coo.dealloc();
  coo.values = values;
  coo.colidx = colidx;
  coo.rowidx = rowidx;

  return true;
}

bool loadMatrixMarketTransposed(const char *file, COO &coo, int pad /*= 1*/)
{
  return loadMatrixMarket_(file, coo, false, true /*transpose*/, pad);
}

bool loadMatrixMarket (const char *file, COO &coo, bool force_symmetric /*=false*/, int pad /*=1*/)
{
  return loadMatrixMarket_(file, coo, force_symmetric, false /*no-transpose*/, pad);
}

} // namespace SpMP
