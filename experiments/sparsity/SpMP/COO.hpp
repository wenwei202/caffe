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

#include "CSR.hpp"

namespace SpMP
{

class COO { // one-based index
public:
  int m;
  int n;
  int nnz;
  int *rowidx;
  int *colidx;
  double *values;
  bool isSymmetric;

  COO();
  ~COO();
  void dealloc();

  void storeMatrixMarket(const char *fileName) const;
};

/**
 * @ret true if succeed
 */
bool loadMatrixMarket (const char *fileName, COO &Acoo, bool force_symmetric = false, int pad = 1);
bool loadMatrixMarketTransposed (const char *fileName, COO &Acoo, int pad = 1);

/**
 * @param createSeparateDiagData true then populate diag and idiag
 */
void dcoo2csr(CSR *Acrs, const COO *Acoo, int outBase = 0, bool createSeparateDiagData = true);
void dcoo2csr(
  int m, int nnz,
  int *rowptr, int *colidx, double *values,
  const int *cooRowidx, const int *cooColidx, const double *cooValues,
  bool sort = true,
  int outBase = 0);

} // namespace SpMP
