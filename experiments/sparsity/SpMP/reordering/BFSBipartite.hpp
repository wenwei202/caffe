#pragma once

#include "../CSR.hpp"

extern "C"
{

void bfsBipartite(SpMP::CSR& A, SpMP::CSR& AT, int *rowPerm, int *rowInversePerm, int *colPerm, int *colInversePerm);

}
