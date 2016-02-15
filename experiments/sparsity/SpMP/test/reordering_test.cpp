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
 * \brief Example of cache-locality optimizing reorderings.
 *
 * \ref "Parallelization of Reordering Algorithms for Bandwidth and Wavefront
 *       Reduction", Karantasis et al., SC 2014
 * \ref "AN OBJECT-ORIENTED ALGORITHMIC LABORATORY FOR ORDERING SPARSEMATRICES",
 *       Kumfert
 * \ref "Fast and Efficient Graph Traversal Algorithms for CPUs: Maximizing
 *       Single-Node Efficiency", Chhugani et al., IPDPS 2012
 * \ref "Multi-core spanning forest algorithms using the disjoint-set data
 *       structure", Patwary et al., IPDPS 2012
 *
 * Expected performance
   (web-Google.mtx can be downloaded from U of Florida matrix collection)
   BW can change run to run.

 In a 18-core Xeon E5-2699 v3 @ 2.3GHz, 56 gbps STREAM BW

OMP_NUM_THREADS=18 KMP_AFFINITY=granularity=fine,compact,1 test/reordering_test web-Google.mtx
/home/jpark103/matrices/web-Google.mtx:::symmetric m=916428 n=916428 nnz=8644102
original bandwidth 915881
SpMV BW   17.07 gbps
BFS reordering
Constructing permutation takes 0.0301461 (1.15 gbps)
Permute takes 0.0295999 (3.50 gbps)
Permuted bandwidth 557632
SpMV BW   37.77 gbps
RCM reordering w/o source selection heuristic
Constructing permutation takes 0.0886319 (0.39 gbps)
Permute takes 0.0256741 (4.04 gbps)
Permuted bandwidth 321046
SpMV BW   43.52 gbps
RCM reordering
Constructing permutation takes 0.143199 (0.24 gbps)
Permute takes 0.0248771 (4.17 gbps)
Permuted bandwidth 330214
SpMV BW   41.32 gbps

 */
 
#include <omp.h>
#include <mkl.h>

#include "../Utils.hpp"

#include "test.hpp"

typedef enum
{
  BFS = 0,
  RCM_WO_SOURCE_SELECTION,
  RCM,
} Option;

int main(int argc, char **argv)
{
  if (argc < 2) {
    fprintf(stderr, "Usage: reordering_test matrix_in_matrix_market_format\n");
    return -1;
  }

  CSR *A = new CSR(argv[1], 0, true /* force-symmetric */);
  int nnz = A->getNnz();
  double flops = 2*nnz;
  double bytes = (double)(sizeof(double) + sizeof(int))*nnz + sizeof(double)*(A->m + A->n);
  printf("m = %d nnz = %d %f bytes = %f\n", A->m, nnz, (double)nnz/A->m, bytes);

  printf("original bandwidth %d\n", A->getBandwidth());

  double *x = MALLOC(double, A->m);
  double *y = MALLOC(double, A->m);

  // allocate a large buffer to flush out cache
  bufToFlushLlc = (double *)_mm_malloc(LLC_CAPACITY, 64);

  const int REPEAT = 128;
  double times[REPEAT];

  for (int i = 0; i < REPEAT; ++i) {
    for (int j = 0; j < 16; ++j) flushLlc();

    double t = omp_get_wtime();
    A->multiplyWithVector(y, x);
    times[i] = omp_get_wtime() - t;
  }
  correctnessCheck(A, y);

  printf("SpMV BW");
  printEfficiency(times, REPEAT, flops, bytes);

#ifdef MKL
  for (int i = 0; i < REPEAT; ++i) {
    for (int j = 0; j < 16; ++j) flushLlc();

    double t = omp_get_wtime();
    mkl_cspblas_dcsrgemv(
      "N", &A->m, A->values, A->rowptr, A->colidx, x, y);
    times[i] = omp_get_wtime() - t;
  }
  correctnessCheck(A, y);

  printf("MKL SpMV BW");
  printEfficiency(times, REPEAT, flops, bytes);
#endif

  int *perm = MALLOC(int, A->m);
  int *inversePerm = MALLOC(int, A->m);

  for (int o = BFS; o <= RCM; ++o) {
    Option option = (Option)o;

    switch (option) {
    case BFS:
      printf("\nBFS reordering\n");
      break;
    case RCM_WO_SOURCE_SELECTION:
      printf("\nRCM reordering w/o source selection heuristic\n");
      break;
    case RCM:
      printf("\nRCM reordering\n");
      break;
    default: assert(false); break;
    }

    double t = -omp_get_wtime();
    switch (option) {
    case BFS:
      A->getBFSPermutation(perm, inversePerm);
      break;
    case RCM_WO_SOURCE_SELECTION:
      A->getRCMPermutation(perm, inversePerm, false);
      break;
    case RCM:
      A->getRCMPermutation(perm, inversePerm);
      break;
    default: assert(false); break;
    }
    t += omp_get_wtime();

    printf(
      "Constructing permutation takes %g (%.2f gbps)\n",
      t, nnz*4/t/1e9);

    isPerm(perm, A->m);
    isPerm(inversePerm, A->m);

    t = -omp_get_wtime();
    CSR *APerm = A->permute(perm, inversePerm);
    t += omp_get_wtime();

    printf("Permute takes %g (%.2f gbps)\n", t, bytes/t/1e9);
    printf("Permuted bandwidth %d\n", APerm->getBandwidth());

    for (int i = 0; i < REPEAT; ++i) {
      for (int j = 0; j < 16; ++j) flushLlc();

      t = omp_get_wtime();
      APerm->multiplyWithVector(y, x);
      times[i] = omp_get_wtime() - t;
    }
    printf("SpMV BW");
    printEfficiency(times, REPEAT, flops, bytes);

    delete APerm;
  }

  FREE(x);
  FREE(y);

  delete A;
}
