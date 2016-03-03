#include <cfloat>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sys/time.h>

#include <string>
#include <algorithm>

#include "Utils.hpp"

using namespace std;

namespace SpMP
{

static const double DEFAULT_CPU_FREQ = 3.33e9;
double get_cpu_freq()
{
  static double freq = DBL_MAX;
  if (DBL_MAX == freq) {
    volatile double a = rand()%1024, b = rand()%1024;
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    unsigned long long t1 = __rdtsc();
    for (size_t i = 0; i < 1024L*1024; i++) {
      a += a*b + b/a;
    }
    unsigned long long dt = __rdtsc() - t1;
    gettimeofday(&tv2, NULL);
    freq = dt/((tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec)/1.e6);
  }

  return freq;
}

void getSimpleThreadPartition(int* begin, int *end, int n)
{
  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();

  int n_per_thread = (n + nthreads - 1)/nthreads;

  *begin = std::min(n_per_thread*tid, n);
  *end = std::min(*begin + n_per_thread, n);
}

void getLoadBalancedPartition(int *begin, int *end, const int *prefixSum, int n)
{
  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();

  int base = prefixSum[0];
  int total_work = prefixSum[n] - base;
  int work_per_thread = (total_work + nthreads - 1)/nthreads;

  *begin = tid == 0 ? 0 : lower_bound(prefixSum, prefixSum + n, work_per_thread*tid + base) - prefixSum;
  *end = tid == nthreads - 1 ? n : lower_bound(prefixSum, prefixSum + n, work_per_thread*(tid + 1) + base) - prefixSum;

  assert(*begin <= *end);
  assert(*begin >= 0 && *begin <= n);
  assert(*end >= 0 && *end <= n);
}

void getInversePerm(int *inversePerm, const int *perm, int n)
{
#pragma omp parallel for
#pragma simd
  for (int i = 0; i < n; ++i) {
    inversePerm[perm[i]] = i;
  }
}

bool isPerm(const int *perm, int n)
{
  int *temp = new int[n];
  memcpy(temp, perm, sizeof(int)*n);
  sort(temp, temp + n);
  int *last = unique(temp, temp + n);
  if (last != temp + n) {
    memcpy(temp, perm, sizeof(int)*n);
    sort(temp, temp + n);

    for (int i = 0; i < n; ++i) {
      if (temp[i] == i - 1) {
        printf("%d duplicated\n", i - 1);
        assert(false);
        return false;
      }
      else if (temp[i] != i) {
        printf("%d missed\n", i);
        assert(false);
        return false;
      }
    }
  }
  delete[] temp;
  return true;
}

bool isInversePerm(const int *perm, const int *inversePerm, int len)
{
  for (int i = 0; i < len; ++i) {
    if (inversePerm[perm[i]] != i) return false;
  }
  return true;
}

template<class T>
void CopyVector(T *dst, const T *src, int len)
{
#pragma omp parallel for
  for (int i = 0; i < len; ++i) {
    dst[i] = src[i];
  }
}

template<class T>
void reorderVectorOutOfPlace_(T *dst, const T *src, const int *perm, int len)
{
  if (perm) {
#pragma omp parallel for
    for (int i = 0; i < len; ++i) {
      assert(perm[i] >= 0 && perm[i] < len);
      dst[perm[i]] = src[i];
    }
  }
  else {
    CopyVector(dst, src, len);
  }
}

void reorderVectorOutOfPlace(double *dst, const double *src, const int *perm, int len)
{
  return reorderVectorOutOfPlace_(dst, src, perm, len);
}

void reorderVectorOutOfPlace(int *dst, const int *src, const int *perm, int len)
{
  return reorderVectorOutOfPlace_(dst, src, perm, len);
}

template<class T>
void reorderVectorOutOfPlaceWithInversePerm_(T *dst, const T *src, const int *inversePerm, int len)
{
  if (inversePerm) {
#pragma omp parallel for
    for (int i = 0; i < len; ++i) {
      assert(inversePerm[i] >= 0 && inversePerm[i] < len);
      dst[i] = src[inversePerm[i]];
    }
  }
  else {
    CopyVector(dst, src, len);
  }
}

void reorderVectorOutOfPlaceWithInversePerm(double *dst, const double *src, const int *inversePerm, int len)
{
  return reorderVectorOutOfPlaceWithInversePerm_(dst, src, inversePerm, len);
}

void reorderVectorOutOfPlaceWithInversePerm(float *dst, const float *src, const int *inversePerm, int len)
{
  return reorderVectorOutOfPlaceWithInversePerm_(dst, src, inversePerm, len);
}

void reorderVectorOutOfPlaceWithInversePerm(int *dst, const int *src, const int *inversePerm, int len)
{
  return reorderVectorOutOfPlaceWithInversePerm_(dst, src, inversePerm, len);
}

double *getReorderVector(const double *v, const int *perm, int len)
{
  double *ret = MALLOC(double, len);

  reorderVectorOutOfPlace(ret, v, perm, len);

  return ret;
}

double *getReorderVectorWithInversePerm(const double *v, const int *perm, int len)
{
  double *ret = MALLOC(double, len);

  reorderVectorOutOfPlaceWithInversePerm(ret, v, perm, len);

  return ret;
}

void reorderVector(double *v, double *tmp, const int *perm, int len)
{
  if (!perm) return;

  reorderVectorOutOfPlace(tmp, v, perm, len);
  CopyVector(v, tmp, len);
}

void reorderVectorWithInversePerm(double *v, double *tmp, const int *inversePerm, int len)
{
  if (!inversePerm) return;

  reorderVectorOutOfPlaceWithInversePerm(tmp, v, inversePerm, len);
  CopyVector(v, tmp, len);
}

void reorderVector(double *v, const int *perm, int len)
{
  double *tmp = MALLOC(double, len);
  reorderVector(v, tmp, perm, len);
  FREE(tmp);
}

void reorderVectorWithInversePerm(double *v, const int *perm, int len)
{
  double *tmp = MALLOC(double, len);
  reorderVectorWithInversePerm(v, tmp, perm, len);
  FREE(tmp);
}

} // namespace SpMP
