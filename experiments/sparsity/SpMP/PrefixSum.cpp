#include "Utils.hpp"

namespace SpMP
{

void prefixSum(int *in_out, int *sum, int *workspace)
{
  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();

  workspace[tid + 1] = *in_out;

#pragma omp barrier
#pragma omp master
  {
    workspace[0] = 0;
    int i;
    for (i = 1; i < nthreads; i++) {
      workspace[i + 1] += workspace[i];
    }
    *sum = workspace[nthreads];
  }
#pragma omp barrier

  *in_out = workspace[tid];
}

void prefixSumMultiple(int *in_out, int *sum, int n, int *workspace)
{
  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();

  int i;
  for (i = 0; i < n; i++) {
    workspace[(tid + 1)*n + i] = in_out[i];
  }

#pragma omp barrier
#pragma omp master
  {
    for (i = 0; i < n; i++) {
      workspace[i] = 0;
    }

    int t;
    // assuming n is not so big, we don't parallelize this loop
    for (t = 1; t < nthreads; t++)
    {
      for (i = 0; i < n; i++) {
        workspace[(t + 1)*n + i] += workspace[t*n + i];
      }
    }

    for (i = 0; i < n; i++) {
      sum[i] = workspace[nthreads*n + i];
    }
  }
#pragma omp barrier

  for (i = 0; i < n; i++) {
    in_out[i] = workspace[tid*n + i];
  }
}

} // namespace SpMP
