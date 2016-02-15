#include <cstdio>

#include "Utils.hpp"
#include "Vector.hpp"
#include "mm_io.h"

namespace SpMP
{

bool loadVectorMatrixMarket(const char *fileName, double **v, int *m, int *n)
{
  FILE *fp = fopen(fileName, "r");
  if (!fp) {
    fprintf(stderr, "Failed to open file %s\n", fileName);
    return false;
  }

  MM_typecode matcode;
  if (mm_read_banner(fp, &matcode) != 0) {
    fprintf(stderr, "Error: could not process Matrix Market banner.\n");
    fclose(fp);
    return false;
  }

  if (!mm_is_valid(matcode)) {
    fprintf(stderr, "Error: invalid Matrix Market banner.\n");
    fclose(fp);
    return false;
  }

  if (!mm_is_array(matcode)) {
    fprintf(stderr, "Error: only support arrays.\n");
    fclose(fp);
    return false;
  }

  if (mm_read_mtx_array_size(fp, m, n) != 0) {
    fprintf(stderr, "Error: could not read array size.\n");
    fclose(fp);
    return false;
  }

  *v = MALLOC(double, (*m)*(*n));

  for (int i = 0; i < *m; i++) {
    for (int j = 0; j < *n - 1; j++) {
      if (fscanf(fp, "%lg", *v + i*(*n) + j) != 1) {
        fprintf(stderr, "Error: premature EOF.\n");
        FREE(*v);
        fclose(fp);
        return false;
      }
    }
    if (fscanf(fp, "%lg\n", *v + i*(*n) + (*n) - 1) != 1) {
      fprintf(stderr, "Error: premature EOF.\n");
      FREE(*v);
      fclose(fp);
      return false;
    }
  }

  fclose(fp);
  return true;
}

} // namespace SpMP
