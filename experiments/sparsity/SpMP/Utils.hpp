#pragma once

#include <vector>
#include <cassert>
#include <math.h>
#include <malloc.h>
#ifndef __INTEL_COMPILER
#include <x86intrin.h>
#endif

#include <omp.h>

#define _malloc_2M(X) \
  mmap64(NULL, (X + 4096), PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_SHARED|MAP_HUGETLB|MAP_POPULATE, -1, 0)

#define FREE(x) { if (x) _mm_free(x); x = NULL; }
#define MALLOC(type, len) (type *)_mm_malloc(sizeof(type)*(len), 64)

namespace SpMP
{

/**
 * Measure CPU frequency by __rdtsc a compute intensive loop.
 */
double get_cpu_freq();

template<typename U, typename T>
bool operator<(const std::pair<U, T>& a, const std::pair<U, T>& b)
{
  if (a.first != b.first) {
    return a.first < b.first;
  }
  else {
    return a.second < b.second;
  }
}

/**
 * Compare two vectors with a given error margin
 */
template<class T>
bool correctnessCheck(
  const T *expected, const T *actual,
  int n,
  double tol = 1e-5, double ignoreSmallerThanThis = 0) {
  for (int i = 0; i < n; ++i) {
    if (fabs(expected[i] - actual[i])/fabs(expected[i]) >= tol &&
      (fabs(expected[i]) >= ignoreSmallerThanThis ||
        fabs(actual[i]) >= ignoreSmallerThanThis)) {
      printf("Error at %d expected %g actual %g\n", i, expected[i], actual[i]);
      assert(false);
      return false;
    }
  }
  return true;
}

/**
 * @note must be called inside an omp region
 */
void getSimpleThreadPartition(int* begin, int *end, int n);

/**
 * Get a load balanced partition so that each thread can work on
 * the range of begin-end where prefixSum[end] - prefixSum[begin]
 * is similar among threads.
 * For example, prefixSum can be rowptr of a CSR matrix and n can be
 * the number of rows. Then, each thread will work on similar number
 * of non-zeros.
 *
 * @params prefixSum monotonically increasing array with length n + 1
 *
 * @note must be called inside an omp region
 */
void getLoadBalancedPartition(int *begin, int *end, const int *prefixSum, int n);

void getInversePerm(int *inversePerm, const int *perm, int n);

/**
 * Let x_i be the input of ith thread.
 * The output of ith thread y_i = x_0 + x_1 + ... + x_{i-1}
 * Additionally, sum = x_0 + x_1 + ... + x_{nthreads - 1}
 * Note that always y_0 = 0
 *
 * @param workspace at least with length (nthreads + 1)
 *
 * @note must be called inside an omp region
 */
void prefixSum(int *inOut, int *sum, int *workspace);
/**
 * workspace[n*tid:n*(tid+1)-1] contains results for tid
 *
 * @param workspace at least with length n*(nthreads + 1)
 *
 * @note must be called inside an omp region
 */
void prefixSumMultiple(int *inOut, int *sum, int n, int *workspace);

/**
 * @return true if perm array is a permutation
 */
bool isPerm(const int *perm, int n);
bool isInversePerm(const int *perm, const int *inversePerm, int len);

/**
 * dst = perm(src)
 */
void reorderVectorOutOfPlace(double *dst, const double *src, const int *perm, int len);
void reorderVectorOutOfPlaceWithInversePerm(double *dst, const double *src, const int *inversePerm, int len);
void reorderVectorOutOfPlaceWithInversePerm(float *dst, const float *src, const int *inversePerm, int len);

void reorderVectorOutOfPlace(int *dst, const int *src, const int *perm, int len);
void reorderVectorOutOfPlaceWithInversePerm(int *dst, const int *src, const int *inversePerm, int len);

/**
 * @return perm(v)
 */
double *getReorderVector(const double *v, const int *perm, int len);
double *getReorderVectorWithInversePerm(const double *v, const int *perm, int len);

/**
 * v = perm(v), use tmp as a temporal buffer
 */
void reorderVector(double *v, double *tmp, const int *perm, int len);
void reorderVectorWithInversePerm(double *v, double *tmp, const int *inversePerm, int len);

/**
 * v = perm(v), a temp buffer will be allocated internally
 */
void reorderVector(double *v, const int *perm, int len);
void reorderVectorWithInversePerm(double *v, const int *inversePerm, int len);

template<class T>
void copyVector(T *out, const T *in, int len)
{
#pragma omp parallel for
  for (int i = 0; i < len; ++i) {
    out[i] = in[i];
  }
}

#define USE_LARGE_PAGE
#ifdef USE_LARGE_PAGE
#include <assert.h>
#include <stdlib.h>
#include <sys/mman.h>
#define HUGE_PAGE_SIZE (2 * 1024 * 1024)
#define ALIGN_TO_PAGE_SIZE(x) \
(((x) + HUGE_PAGE_SIZE - 1) / HUGE_PAGE_SIZE * HUGE_PAGE_SIZE)

#ifndef MAP_HUGETLB
# define MAP_HUGETLB  0x40000
#endif

inline void *malloc_huge_pages(size_t size)
{
// Use 1 extra page to store allocation metadata
// (libhugetlbfs is more efficient in this regard)
size_t real_size = ALIGN_TO_PAGE_SIZE(size + HUGE_PAGE_SIZE);
char *ptr = (char *)mmap(NULL, real_size, PROT_READ | PROT_WRITE,
MAP_PRIVATE | MAP_ANONYMOUS |
MAP_POPULATE | MAP_HUGETLB, -1, 0);
if (ptr == MAP_FAILED) {
// The mmap() call failed. Try to malloc instead
posix_memalign((void **)&ptr, 4096, real_size);
if (ptr == NULL) return NULL;
real_size = 0;
}
// Save real_size since mmunmap() requires a size parameter
*((size_t *)ptr) = real_size;
// Skip the page with metadata
return ptr + HUGE_PAGE_SIZE;
}
inline void free_huge_pages(void *ptr)
{
if (ptr == NULL) return;
// Jump back to the page with metadata
void *real_ptr = (char *)ptr - HUGE_PAGE_SIZE;
// Read the original allocation size
size_t real_size = *((size_t *)real_ptr);
assert(real_size % HUGE_PAGE_SIZE == 0);
if (real_size != 0)
// The memory was allocated via mmap()
// and must be deallocated via munmap()
munmap(real_ptr, real_size);
else
// The memory was allocated via malloc()
// and must be deallocated via free()
free(real_ptr);
}
#undef USE_LARGE_PAGE
#endif // USE_LARGE_PAGE

} // namespace SpMP
