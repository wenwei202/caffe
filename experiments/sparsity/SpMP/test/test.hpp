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

#include <algorithm>

#include "../CSR.hpp"

using namespace std;
using namespace SpMP;

bool correctnessCheck(CSR *A, double *y)
{
  static double *yt = NULL;
  if (NULL == yt) {
    yt = new double[A->m];

    copy(y, y + A->m, yt);

    return true;
  }
  else {
    return correctnessCheck<double>(yt, y, A->m, 1e-8);
  }
}

static void printEfficiency(
  double *times, int REPEAT, double flop, double byte)
{
  sort(times, times + REPEAT);

  double t = times[REPEAT/2];

  printf(
    "%7.2f gflops %7.2f gbps\n",
    flop/t/1e9, byte/t/1e9);
}

static const size_t LLC_CAPACITY = 32*1024*1024;
static const double *bufToFlushLlc = NULL;

void flushLlc()
{
  double sum = 0;
#pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < LLC_CAPACITY/sizeof(bufToFlushLlc[0]); ++i) {
    sum += bufToFlushLlc[i];
  }
  FILE *fp = fopen("/dev/null", "w");
  fprintf(fp, "%f\n", sum);
  fclose(fp);
}

void initializeX(double *x, int n) {
#pragma omp parallel for
  for (int i = 0; i < n; ++i) x[i] = n - i;
}

typedef enum
{
  REFERENCE = 0,
  BARRIER,
  P2P,
  P2P_WITH_TRANSITIVE_REDUCTION,
} SynchronizationOption;

#define ADJUST_FOR_BASE \
  int base = A.getBase(); \
  const int *rowptr = A.rowptr - base; \
  const int *colidx = A.colidx - base; \
  const double *values = A.values - base; \
  const double *idiag = A.idiag - base; \
  y -= base; \
  b -= base;
