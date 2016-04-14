#!/usr/bin/env python

import sys, re

sparsity_pattern = re.compile("^.*\.weight: .*\(([\d.]+) nnz-sparsity ([\d.]+) col-sparsity ([\d.]+) row-sparsity ([\d.]+) kernel-sparsity.*")
dense_pattern = re.compile("^dense_parbatch:\s+[\d.]+\s+dense_gflops\s+([\d.]+)\s+dense_gflops.*")
compress_pattern = re.compile("^compressed_parbatch:\s+[\d.]+\s+dense_gflops\s+([\d.]+)\s+dense_gflops.*")
sparse_pattern = re.compile("^conv_parbatch:\s+[\d.]+\s+dense_gflops\s+([\d.]+)\s+dense_gflops.*")

nnz_sparsities = []
col_sparsities = []
row_sparsities = []
kernel_sparsities = []
dense_gflops = []
compress_gflops = []
sparse_gflops = []

for line in open(sys.argv[1]):
  m = sparsity_pattern.match(line)
  if m:
    nnz_sparsities.append(m.group(1))
    col_sparsities.append(m.group(2))
    row_sparsities.append(m.group(3))
    kernel_sparsities.append(m.group(4))
  m = dense_pattern.match(line)
  if m:
    dense_gflops.append(m.group(1))
  m = compress_pattern.match(line)
  if m:
    compress_gflops.append(m.group(1))
  m = sparse_pattern.match(line)
  if m:
    sparse_gflops.append(m.group(1))

assert len(nnz_sparsities) == len(col_sparsities)
assert len(nnz_sparsities) == len(row_sparsities)
assert len(nnz_sparsities) == len(kernel_sparsities)
for i in range(len(nnz_sparsities)):
  print nnz_sparsities[i], col_sparsities[i], row_sparsities[i], kernel_sparsities[i]

assert len(dense_gflops) == len(compress_gflops)
assert len(dense_gflops) == len(sparse_gflops)
for i in range(len(dense_gflops)):
  print dense_gflops[i], compress_gflops[i], sparse_gflops[i]
