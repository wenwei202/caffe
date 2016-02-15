#include "../CSR.hpp"
#include "../mm_io.h"
#include "../synk/barrier.hpp"

using namespace SpMP;

synk::Barrier *bar;

int main(int argc, const char *argv[])
{
  if (argc < 3) {
    printf("Usage: mtx2bin matrix_market_file_name petsc_bin_file_name\n");
    return -1;
  }

  FILE *fp = fopen(argv[1], "r");

  MM_typecode matcode;
  int nrows, ncols;
  int read_banner_ret = mm_read_banner(fp, &matcode);
  int is_valid_ret = mm_is_valid(matcode);
  int is_array_ret = mm_is_array(matcode);
  int read_mtx_array_size_ret = mm_read_mtx_array_size(fp, &nrows, &ncols);

  if (0 == read_banner_ret && 0 == read_mtx_array_size_ret &&
    is_valid_ret && is_array_ret) {
    printf("m=%d\n", nrows);
    double *a = new double[nrows];
    for (int i = 0; i < nrows; ++i) {
      double v;
      fscanf(fp, "%lg\n", &v);
      a[i] = v;
    }
    fclose(fp); fp = NULL;

    fp = fopen(argv[2], "w");
    if (NULL == fp) {
      fprintf(stderr, "Failed to open %s\n", argv[2]);
      return -1;
    }

    fwrite(&nrows, sizeof(nrows), 1, fp);
    fwrite(a, sizeof(a[0]), nrows, fp);

    fclose(fp); fp = NULL;
  }
  else {
    fclose(fp); fp = NULL;

    CSR A(argv[1]);
    printf("m=%d n=%d nnz=%d\n", A.m, A.n, A.rowptr[A.m]);
    A.storeBin(argv[2]);

//#define CHECK_CORRECTNESS
#ifdef CHECK_CORRECTNESS
    CSR B(argv[2]);
    assert(A.equals(B, true));
#endif
  }

  return 0;
}
