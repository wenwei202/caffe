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

#pragma once

#include <cstdlib>
#include <vector>
#include <string>

#include "MemoryPool.hpp"
#include "Utils.hpp"

namespace SpMP
{

class CSR {
public:
  int m;
  int n;
  int *rowptr;
  int *colidx;
  double *values;

  double *idiag; /**< inverse of diagonal elements */
  double *diag;
  int *diagptr;
  int *extptr; // points to the beginning of non-local columns (when MPI is used)

  CSR();

  // Following two constructors will make CSR own the data
  CSR(const char *file, int base = 0, bool forceSymmetric = false, int pad = 1);
  CSR(int m, int n, int nnz);
  CSR(const CSR& A);

  // Following constructor will make CSR does not own the data
  CSR(int m, int n, int *rowptr, int *colidx, double *values);
 
  ~CSR();

  void storeMatrixMarket(const char *fileName) const;
  /**
   * Load PETSc bin format
   */
  void loadBin(const char *fileName, int base = 0);
  /**
   * Load PETSc bin format
   */
  void storeBin(const char *fileName);

  /**
   * permutes current matrix based on permutation vector "perm"
   * and return new permuted matrix
   *
   * @param sort true if we want to sort non-zeros of each row based on colidx
   */
  CSR *permute(const int *columnPerm, const int *rowInversePerm, bool sort = false) const;
  /**
   * permutes rows but not columns
   */
  CSR *permuteRows(const int *inversePerm) const;
  /**
   * just permute rowptr
   */
  CSR *permuteRowptr(const int *inversePerm) const;
  void permuteRowptr(CSR *out, const int *inversePerm) const;

  void permuteColsInPlace(const int *perm);
  void permuteInPlaceIgnoringExtptr(const int *perm);

  /**
   * assuming rowptr is permuted, permute the remaining (colidx and values)
   *
   * @param sort true if we want to sort non-zeros of each row based on colidx
   */
  void permuteMain(
    CSR *out, const int *columnPerm, const int *rowInversePerm,
    bool sort = false) const;
  void permuteRowsMain(
    CSR *out, const int *inversePerm) const;

  /**
   * Compute w = alpha*A*x + beta*y + gamma
   * where A is this matrix
   */
  void multiplyWithVector(
    double *w,
    double alpha, const double *x, double beta, const double *y, double gamma)
    const;

  /**
   * Compute w = A*x
   */
  void multiplyWithVector(double *w, const double *x) const;

  /**
   * Compute W = alpha*A*X + beta*Y + gamma.
   * W, X, and Y are dense matrices with width k.
   */
  void multiplyWithDenseMatrix(
    double *W, int k, int wRowStride, int wColumnStride,
    double alpha,
    const double *X, int xRowStride, int xColumnStride,
    double beta, const double *Y, int yRowStride, int yColumnStride,
    double gamma) const;

  /**
   * Compute W = A*X.
   * W and X are dense matrices with width k, and are in column major
   */
  void multiplyWithDenseMatrix(double *W, int k, const double *X) const;

  /**
   * get reverse Cuthill Mckee permutation that tends to reduce the bandwidth
   *
   * @note only works for a symmetric matrix
   *
   * @param pseudoDiameterSourceSelection true to use heurstic of using a source
   *                                      in a pseudo diameter.
   *                                      Further reduce diameter at the expense
   *                                      of more time on finding permutation.
   */
  void getRCMPermutation(int *perm, int *inversePerm, bool pseudoDiameterSourceSelection = true);
  void getBFSPermutation(int *perm, int *inversePerm);

  CSR *transpose() const;

  bool isSymmetric(bool checkValues = true, bool printFirstNonSymmetry = false) const;

  void make0BasedIndexing();
  void make1BasedIndexing();

  /**
   * Precompute idiag to speedup triangular solver or GS
   */
  void computeInverseDiag();

  /**
   * Need to do this before permutation.
   * constructDiagPtr after permutation won't be correct
   */
  void constructDiagPtr();

  void alloc(int m, int nnz, bool createSeparateDiagData = true);
  void dealloc();

  bool useMemoryPool_() const;

  int getBandwidth() const;
  double getAverageWidth(bool sorted = false) const;
  bool equals(const CSR& A, bool print = false) const;
  int getNnz() const { return rowptr[m] - getBase(); }
  int getBase() const { return rowptr[0]; }

  void print() const;

  template<class T> T *allocate_(size_t n) const
  {
    if (useMemoryPool_()) {
      return MemoryPool::getSingleton()->allocate<T>(n);
    }
    else {
      return MALLOC(T, n);
    }
  }

private:
  bool ownData_;
}; // CSR

void generate3D27PtLaplacian(CSR *A, int nx, int ny, int nz, int base = 0);
void generate3D27PtLaplacian(CSR *A, int n, int base = 0);

void splitLU(CSR& A, CSR *L, CSR *U);
bool getSymmetricNnzPattern(
  const CSR *A, int **symRowPtr, int **symDiagPtr, int **symExtPtr, int **symColIdx);

} // namespace SpMP
