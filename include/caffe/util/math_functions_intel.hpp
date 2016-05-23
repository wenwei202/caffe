#ifndef CAFFE_UTIL_MATH_FUNCTIONS_INTEL_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_INTEL_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

#include "synk/barrier.hpp"

extern synk::Barrier *barriers[256];

namespace caffe {

/**
 * Direct convolution with dense tensors
 */
template <typename Dtype>
void caffe_cpu_dconv(
    // input features
    const Dtype *input,
    int in_channels, int height, int width,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    // weights
    const Dtype *weight,
    int kernel_h, int kernel_w,
    // bias (for the case when bias is fused with convolution)
    const Dtype *bias, const Dtype *bias_multiplier,
    // pooling (for the case when pooling is fused with convolution)
    Dtype *pool_top, int *mask,
    // output features
    Dtype *output,
    int out_channels);

template <typename Dtype>
void caffe_cpu_sconv(
    // input features
    const Dtype *input_padded, const Dtype *input, int in_channels,
    int height, int width,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    // weights
    const int *rowptr, const int *colidx, const Dtype *values,
    int kernel_h, int kernel_w,
    const int **rowptr_blocked, const int **colidx_blocked, const Dtype **values_blocked,
    int ncolblocks,
//    const int *blockptr, const int *kidx, const float *values_colmajor,
//    Dtype *input_scratch, Dtype *output_colmajor_scratch,
    // bias (for the case when bias is fused with convolution)
    const Dtype *bias, const Dtype *bias_multiplier,
    // pooling (for the case when pooling is fused with convolution)
    Dtype *pool_top, int *mask,
    // output features
    Dtype *output,
    int out_channels,
    float *output_scratch, int col_major_ic_block,
    const Dtype *input_aligned,
    const int *weight_rowptr_split_,
    const int *weight_colidx_split_,
    const Dtype *weight_values_split_);

}  // namespace caffe

#endif  // CAFFE_UTIL_MATH_FUNCTIONS_INTEL_H_
