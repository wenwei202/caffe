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

#include "CSR.hpp"

namespace SpMP
{

void generate3D27PtLaplacian(CSR *A, int nx, int ny, int nz, int base /*=0*/) {
  A->dealloc();

  A->m = A->n = nx*ny*nz;
  A->rowptr = MALLOC(int, A->m + 1);
  A->diagptr = MALLOC(int, A->m);
  A->idiag = MALLOC(double, A->m);
  A->diag = MALLOC(double, A->m);
  A->rowptr[0] = 0;

  int idx = 0;
//#define LOWER_TRIANGULAR
#ifdef LOWER_TRIANGULAR
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        int row = (iz*ny + iy)*nx + ix;

        for (int sz = -1; sz <= 1; ++sz) {
          if (iz + sz < 0 || iz + sz >= nz) continue;
          for (int sy = -1; sy <= 1; ++sy) {
            if (iy + sy < 0 || iy + sy >= ny) continue;
            for (int sx = -1; sx <= 1; ++sx) {
              if (ix + sx < 0 || ix + sx >= nx) continue;
              int col = row + (sz*ny + sy)*nx + sx;
              if (col > row) continue;
              ++idx;
            } //sx
          } // sy
        } // sz

        A->rowptr[row + 1] = idx;
      } // ix
    } // iy
  } // iz
#else
  for (int iz = 0; iz < nz; ++iz) {
    int dz = (iz >= 1 && iz < nz - 1) ? 3 : (nz == 1 ? 1 : 2);
    for (int iy = 0; iy < ny; ++iy) {
      int dy = ((iy >= 1 && iy < ny - 1) ? 3 : (nz == 1 ? 1 : 2))*dz;
      for (int ix = 0; ix < nx; ++ix) {
        int row = (iz*ny + iy)*nx + ix;

        int dx = (ix >= 1 && ix < nx - 1) ? 3 : (nz == 1 ? 1 : 2);
        idx += dx*dy;
        A->rowptr[row + 1] = idx;
      } // ix
    } // iy
  } // iz
#endif

  int nnz = idx;
  A->colidx = MALLOC(int, nnz);
  A->values = MALLOC(double, nnz);

#pragma omp parallel for collapse(2)
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      int sz_begin = iz <= 0 ? 0 : -1;
      int sz_end = iz >= nz - 1 ? 0 : 1;

      int sy_begin = iy <= 0 ? 0 : -1;
      int sy_end = iy >= ny - 1 ? 0 : 1;
      for (int ix = 0; ix < nx; ++ix) {
        int sx_begin = ix <= 0 ? 0 : -1;
        int sx_end = ix >= nx - 1 ? 0 : 1;

        int row = (iz*ny + iy)*nx + ix;
        int idx = A->rowptr[row];

        for (int sz = sz_begin; sz <= sz_end; ++sz) {
          for (int sy = sy_begin; sy <= sy_end; ++sy) {
            for (int sx = sx_begin; sx <= sx_end; ++sx) {
              int col = row + (sz*ny + sy)*nx + sx;
#ifdef LOWER_TRIANGULAR
              if (col > row) continue;
#endif
              if (col == row) {
                A->diag[row] = 26.0;
                A->values[idx] = A->diag[row];
                A->idiag[row] = 1/A->diag[row];
                A->diagptr[row] = idx;
              }
              else {
                A->values[idx] = -1;
              }
              A->colidx[idx] = col;
              ++idx;
            } //sx
          } // sy
        } // sz
      } // ix
    } // iy
  } // iz

  if (1 == base) {
    A->make1BasedIndexing();
  }
  else {
    assert(0 == base);
  }

  assert(idx == nnz);
}

void generate3D27PtLaplacian(CSR *A, int n, int base /*=0*/) {
  generate3D27PtLaplacian(A, n, n, n, base);
}

} // namespace SpMP
