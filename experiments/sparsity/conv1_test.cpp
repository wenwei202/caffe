/*
 * conv1_test.cpp
 *
 *  Created on: Apr 15, 2016
 *      Author: jpark103
 */

#include <cstdio>
#include <omp.h>
#include <immintrin.h>
#include <vector>
#include <cassert>
#include <cstring>

#include "../../include/caffe/util/conv.hpp"
#include "SpMP/CSR.hpp"

//#define SEP
#ifdef SEP
#include "sampling.h"
#endif
//#define VTUNE
#ifdef VTUNE
#include "ittnotify.h"
#endif

#ifdef SNIPER
#include "sim_api.h"
#endif

using namespace SpMP;
using namespace std;

synk::Barrier *barriers[256];

unsigned long long conv_cycles_of_this_batch[1024*16];

int main(int argc, const char *argv[])
{
#ifdef SNIPER
  const int NBATCH = 1;
#else
  const int NBATCH = 256;
#endif

  int nthreads = omp_get_max_threads();
  int nthread_groups = nthreads;
#ifdef __AVX512F__
  nthread_groups = NTILES;
#else
//  nthread_groups = nthreads/2;
#endif

  assert(nthreads%nthread_groups == 0);
  int nthreads_per_group = nthreads/nthread_groups;
  if (nthread_groups != nthreads) {
    for (int i = 0; i < nthread_groups; ++i) {
      barriers[i] = new synk::Barrier(1, nthreads_per_group);
    }
#pragma omp parallel
    {
      assert(omp_get_num_threads() == nthreads);

      int tid = omp_get_thread_num();
      int gid = tid/nthreads_per_group;
      int tid_in_group = tid%nthreads_per_group;

      barriers[gid]->init(tid_in_group);
    }
  }

  // conv1
//  const int NOUT = 96;
//  const int NIN = 3;
//  const int K = 11;
//  const int WIDTH = 227;
//  const int OUT_WIDTH = 55;
//  const int KERNEL_SIZE_ALIGNED = 128;

  //const float *weight = new float[NOUT * NIN * K * K];
//  const float *weight = new float[NOUT * NIN * KERNEL_SIZE_ALIGNED];
//  const float *input = new float[NBATCH * NIN * WIDTH * WIDTH];
//  float *output = new float[NBATCH * NOUT * OUT_WIDTH * OUT_WIDTH];

  double cpu_freq = get_cpu_freq();
  printf("freq = %g\n", cpu_freq);

  // conv3
  const int NOUT = 384;
  const int NIN = 256;
  const int K = 3;
  const int WIDTH = 13;
  const int WOUT = 13;
  const int PAD = 1;

  CSR *A = new CSR(argv[1]);
  float *values = (float *)_mm_malloc(sizeof(float)*A->getNnz(), 4096);

  int ncolblocks = NIN/COL_BLOCK;

  vector<int *> rowptr_blocked;
  vector<int *> colidx_blocked;
  vector<float *> values_blocked;
  rowptr_blocked.resize(ncolblocks);
  colidx_blocked.resize(ncolblocks);
  values_blocked.resize(ncolblocks);
  std::vector<int> nnzs_of_col_blocks(ncolblocks, 0);

  int *blockptr_colmajor;
  int *kidx_colmajor;
  float *values_colmajor;

  posix_memalign((void **)&blockptr_colmajor, 4096, sizeof(int)*(NIN/COL_MAJOR_IC_BLOCK*NOUT + 1));
  memset(blockptr_colmajor, 0, sizeof(int)*(NIN/COL_MAJOR_IC_BLOCK*NOUT + 1));
  posix_memalign((void **)&kidx_colmajor, 4096, sizeof(int)*A->getNnz());
  posix_memalign((void **)&values_colmajor, 4096, sizeof(float)*A->getNnz());

  for (int out_channel = 0; out_channel < NOUT; ++out_channel) {
    for (int j = A->rowptr[out_channel]; j < A->rowptr[out_channel + 1]; ++j) {
      int col = A->colidx[j];

      int kernel_col = col%K;
      int kernel_row = (col/K)%K;
      int in_channel = col/(K*K);
      assert(in_channel < NIN);

      A->colidx[j] = (in_channel*(WIDTH + PAD) + kernel_row)*(WIDTH + PAD) + kernel_col;
      values[j] = A->values[j];

      int bcol = in_channel/COL_BLOCK;
      nnzs_of_col_blocks[bcol]++;

      int bcol_colmajor = in_channel/COL_MAJOR_IC_BLOCK;
      ++blockptr_colmajor[bcol_colmajor*NOUT + out_channel + 1];
    }
  }

  for (int i = 0; i < ncolblocks; ++i) {
//    rowptr_blocked[i] = (int *)malloc_huge_pages(sizeof(int)*(NOUT + 1));
//    colidx_blocked[i] = (int *)malloc_huge_pages(sizeof(int)*nnzs_of_col_blocks[i]);
//    values_blocked[i] = (float *)malloc_huge_pages(sizeof(float)*nnzs_of_col_blocks[i]);

    posix_memalign((void **)&rowptr_blocked[i], 4096, sizeof(int)*(NOUT + 1));
    posix_memalign((void **)&colidx_blocked[i], 4096, sizeof(int)*nnzs_of_col_blocks[i]);
    posix_memalign((void **)&values_blocked[i], 4096, sizeof(float)*nnzs_of_col_blocks[i]);
    nnzs_of_col_blocks[i] = 0;
    rowptr_blocked[i][0] = 0;
  }

  for (int i = 1; i < NIN/COL_MAJOR_IC_BLOCK*NOUT; ++i) {
    blockptr_colmajor[i + 1] += blockptr_colmajor[i];
  }

  const int SCRATCH_SIZE_PER_IC = (WOUT*WOUT + 15)/16*16;
  for (int out_channel = 0; out_channel < NOUT; ++out_channel) {
    for (int j = A->rowptr[out_channel]; j < A->rowptr[out_channel + 1]; ++j) {
      int c = A->colidx[j];
      int kernel_col = c%(WIDTH + PAD);
      int kernel_row = c/(WIDTH + PAD)%(WIDTH + PAD);
      int in_channel = c/(WIDTH + PAD)/(WIDTH + PAD);
      int bcol = in_channel/COL_BLOCK;

      colidx_blocked[bcol][nnzs_of_col_blocks[bcol]] = c;
      values_blocked[bcol][nnzs_of_col_blocks[bcol]] = values[j];
      nnzs_of_col_blocks[bcol]++;

      int blockid = in_channel/COL_MAJOR_IC_BLOCK*NOUT + out_channel;
      int offset = blockptr_colmajor[blockid];
      kidx_colmajor[offset] = ((in_channel%COL_MAJOR_IC_BLOCK*K + kernel_row)*K + kernel_col)*SCRATCH_SIZE_PER_IC;
      values_colmajor[offset] = values[j];
      ++blockptr_colmajor[blockid];
    }

    for (int i = 0; i < ncolblocks; ++i) {
      rowptr_blocked[i][out_channel + 1] = nnzs_of_col_blocks[i];
    }
  }

  for (int i = NIN/COL_MAJOR_IC_BLOCK*NOUT - 1; i > 0; --i) {
    blockptr_colmajor[i] = blockptr_colmajor[i - 1];
  }
  blockptr_colmajor[0] = 0;
  for (int out_channel = 0; out_channel < NOUT; ++out_channel) {
    int nnz_of_oc = 0;
    for (int i = 0; i < NIN/COL_MAJOR_IC_BLOCK; ++i) {
      nnz_of_oc += blockptr_colmajor[i*NOUT + out_channel + 1] - blockptr_colmajor[i*NOUT + out_channel];
    }
    if (nnz_of_oc != A->rowptr[out_channel + 1] - A->rowptr[out_channel]) {
      printf("oc = %d rowptr[oc+1] - rowptr[oc] expected %d actual %d\n", out_channel, A->rowptr[out_channel + 1] - A->rowptr[out_channel], nnz_of_oc);
    }
  }

  float *input = (float *)_mm_malloc(sizeof(float)*NBATCH*NOUT*(WIDTH + PAD)*(WIDTH + PAD), 4096);
//  float *input = (float *)malloc_huge_pages(sizeof(float)*NBATCH*NOUT*(WIDTH + PAD)*(WIDTH + PAD));
  memset((void *)input, 0, sizeof(float)*NBATCH*NOUT*(WIDTH + PAD)*(WIDTH + PAD));
  float *output = (float *)_mm_malloc(sizeof(float)*NBATCH*NOUT*(WIDTH + PAD)*(WIDTH + PAD), 4096);
//  float *output = (float *)malloc_huge_pages(sizeof(float)*NBATCH*NOUT*(WIDTH + PAD)*(WIDTH + PAD));
  memset((void *)output, 0, sizeof(float)*NBATCH*NOUT*(WIDTH + PAD)*(WIDTH + PAD));
  const float *bias = (float *)_mm_malloc(sizeof(float)*NOUT, 4096);
//  const float *bias = (float *)malloc_huge_pages(sizeof(float)*NOUT);
  memset((void *)bias, 0, sizeof(float)*NOUT);
  const float *bias_multiplier = (float *)_mm_malloc(sizeof(float)*NOUT, 4096);
//  const float *bias_multiplier = (float *)malloc_huge_pages(sizeof(float)*NOUT);
  memset((void *)bias_multiplier, 0, sizeof(float)*NOUT);
  float *scratch = (float *)_mm_malloc(sizeof(float)*OC_BLOCK*WIDTH*16*omp_get_max_threads(), 4096);
  memset((void *)scratch, 0, sizeof(float)*OC_BLOCK*WIDTH*16*omp_get_max_threads());

  float *input_scratch = (float *)_mm_malloc(sizeof(float)*omp_get_max_threads()*COL_MAJOR_IC_BLOCK*K*K*SCRATCH_SIZE_PER_IC, 4096);
  memset((void *)input_scratch, 0, sizeof(float)*omp_get_max_threads()*COL_MAJOR_IC_BLOCK*K*K*SCRATCH_SIZE_PER_IC);

  double t = omp_get_wtime();
  unsigned long long times[omp_get_max_threads()*16];
  for (int i = 0; i < omp_get_max_threads(); ++i) {
    times[i*16] = 0;
    conv_cycles_of_this_batch[i*16] = 0;
  }

#ifdef SNIPER
  const int REPEAT = 2;
#else
  const int REPEAT = 256;
#endif

#ifdef SEP
  VTResumeSampling();
#endif
#ifdef VTUNE
  fprintf(stderr, "__itt_resume\n");
  __itt_resume();
#endif

#ifdef SNIPER
  SimWarmup();
#endif

  for (int j = 0; j < REPEAT; ++j) {
#pragma omp parallel
    {
      int nthreads = omp_get_num_threads();
      int tid = omp_get_thread_num();

#ifdef SNIPER
      if (j == REPEAT - 1) {
        SimRoiStart();

        SimAddAddressRoi(input, sizeof(float)*NIN*(WIDTH + PAD)*(WIDTH + PAD));
        SimSetAddressRoiName(input, "input");

        SimAddAddressRoi(output, sizeof(float)*NOUT*WOUT*WOUT);
        SimSetAddressRoiName(output, "output");

        SimAddAddressRoi(blockptr_colmajor, sizeof(int)*(NIN/COL_MAJOR_IC_BLOCK*NOUT + 1));
        SimSetAddressRoiName(blockptr_colmajor, "blockptr");

        SimAddAddressRoi(kidx_colmajor, sizeof(int)*A->getNnz());
        SimSetAddressRoiName(kidx_colmajor, "kidx");

        SimAddAddressRoi(values_colmajor, sizeof(float)*A->getNnz());
        SimSetAddressRoiName(values_colmajor, "values");

        SimAddAddressRoi(input_scratch, sizeof(float)*omp_get_max_threads()*COL_MAJOR_IC_BLOCK*K*K*SCRATCH_SIZE_PER_IC);
        SimSetAddressRoiName(input_scratch, "scratch");

        SimAddAddressRoi(bias, sizeof(float)*NOUT);
        SimSetAddressRoiName(bias, "bias");
      }
#endif

      int nthread_groups = nthreads;
#ifdef __AVX512F__
      nthread_groups = NTILES;
#endif
      assert(nthreads%nthread_groups == 0);
      int nthreads_per_group = nthreads/nthread_groups;
      int gid = tid/nthreads_per_group;

      int i_per_group = (NBATCH + nthread_groups - 1)/nthread_groups;
      int i_begin = std::min(i_per_group*gid, NBATCH);
      int i_end = std::min(i_begin + i_per_group, NBATCH);

      unsigned long long tt = __rdtsc();

      for (int i = i_begin; i < i_end; ++i) {
        sconv345_ver2(
            input + i*NIN*(WIDTH + PAD)*(WIDTH + PAD), NIN,
            blockptr_colmajor, kidx_colmajor, values_colmajor,
            bias,
            output + i*NOUT*WOUT*WOUT, NOUT,
            input_scratch);
//        sconv345(
//            (j%2 == 0 ? input : output) + i*NIN*(WIDTH + PAD)*(WIDTH + PAD),
//            A->rowptr, A->colidx, values,
//            (const int **)(&rowptr_blocked[0]), (const int **)(&colidx_blocked[0]), (const float **)(&values_blocked[0]),
//            ncolblocks,
//            bias,
//            (j%2 == 0 ? output : input) + i*NOUT*(WIDTH + PAD)*(WIDTH + PAD), NOUT,
//            scratch + tid*OC_BLOCK*WIDTH*16);
      }

#ifdef SNIPER
      if (j == REPEAT - 1) SimRoiEnd();
#endif

      times[tid*16] += __rdtsc() - tt;
    }
  }

#ifdef SEP
  VTPauseSampling();
#endif
#ifdef VTUNE
  __itt_pause();
  fprintf(stderr, "__itt_pause\n");
#endif

  t = omp_get_wtime() - t;

  unsigned long long max_time = 0, max_time2 = 0;
  unsigned long long sum_time = 0;
  for (int i = 0; i < omp_get_max_threads(); ++i) {
    max_time = std::max(max_time, times[i*16]);
    max_time2 = std::max(max_time, conv_cycles_of_this_batch[i*16]);
    sum_time += times[i*16];
  }

  double flops = (double)NOUT*NIN*WIDTH*WIDTH*K*K*2;
  printf("mflops-per-file %g\n", flops/1e6);
  printf("effective-GF/s %g %g\n", flops*REPEAT*NBATCH/t/1e9, flops*REPEAT*NBATCH/(max_time/cpu_freq)/1e6);
  printf("time1 = %g, max_time = %g, avg_time = %g, tt = %g\n", t, max_time/cpu_freq, (double)sum_time/omp_get_max_threads()/cpu_freq, max_time2/cpu_freq);

  return 0;
}
