/*
 * conv.hpp
 *
 *  Created on: Apr 17, 2016
 *      Author: jpark103
 */

#ifndef SRC_CAFFE_LAYERS_CONV_HPP_
#define SRC_CAFFE_LAYERS_CONV_HPP_

#include <immintrin.h>
#include "synk/barrier.hpp"

#ifdef __AVX512F__
#ifdef SNIPER
static const int NTILES = 2; // 1 tile
#else
static const int NTILES = 64; // FIXME - hardcoded for 68c KNL
#endif
#endif

static const int OC_BLOCK = 8;
static const int COL_BLOCK = 32;

static const int COL_MAJOR_IC_BLOCK = 4;
//static const int COL_MAJOR_OC_BLOCK = 64;

extern synk::Barrier *barriers[256];

// input channel fully unrolled
inline void conv1_ver1(const float *weight, const float *input, float *output)
{
  const int M = 96;
  const int WIDTH = 227;
  const int STRIDE = 4;
  const int K = 11;
  const int WOUT = (WIDTH - K)/STRIDE + 1; // 55

  // JSP: AlexNet conv1
  // Input: 3 x 227 x 227 => 201 KB per channel, 604 KB total
  // Output: 96 x 55 x 55 => 12 KB per channel, 1.1 MB total
  // Weight: 96 x 3 x 11 x 11 => 0.5 KB per channel pair, 1.5 KB per output channel, 45 KB per input channel, 136 KB total
  //         No matter what we do, there's no reuse on weight across different channels (only reuse is within a channel pair)
  // FLOPS: 2 x 96 x 3 x 55 x 55 x 11 x 11 = 211 MFLOPS
  //
  // Approach 1: stream output channel
  //   For each output channel, read inputs from LLC (604 KB total) and read weights from memory
  //   Total memory access: input 604 KB, output 1.1 MB, weight 136 KB
  //   Total LLC access: input 96 x 604 KB
  //
  // Approach 2: stream input channel
  //   For each input channel, read outputs from LLC (1.1 MB total) and read weights from memory
  //   Total memory access: input 604 KB, output 1.1 MB, weight 136 KB
  //   Total LLC access: output 2 x 1.1 MB
  //
  // Approach 3: blocking
  //   For each input channel, read 1/8 of outputs from L2 (1.1/8 MB total) and read weights from memory
  //   Total memory access: input 604 KB, output 1.1 MB, weight 136 KB
  //   Total LLC access: input 7 x 604 KB

  const int WOUT_BLOCK = 55;
  const int KERNEL_SIZE_ALIGNED = 128;

  __declspec(aligned(64)) float sum_temp[8];
  int mask1_temp[8] = { -1, 0, 0, 0, 0, 0, 0, 0 };
  __m256i mask1 = _mm256_load_si256((__m256i *)mask1_temp);

  for (int out_channel = 0; out_channel < M; ++out_channel) {
    for (int output_row = 0; output_row < WOUT; ++output_row) {
      for (int output_col = 0; output_col < WOUT; ++output_col) {
        const float *input_temp1 = input + output_col * STRIDE;
        const float *input_temp2 = input_temp1 + WIDTH * WIDTH;
        const float *input_temp3 = input_temp1 + 2 * WIDTH * WIDTH;

        const float *weight_temp1 = weight + 3 * out_channel * KERNEL_SIZE_ALIGNED;
        const float *weight_temp2 = weight_temp1 + KERNEL_SIZE_ALIGNED;
        const float *weight_temp3 = weight_temp1 + 2 * KERNEL_SIZE_ALIGNED;

        const int PREFETCH_DISTANCE = 16;

        _mm_prefetch((const char *)(input_temp1 + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 2*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 3*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 4*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 5*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 6*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 7*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 8*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 9*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 10*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);

        __m256 sum_v[3];
        sum_v[0] = _mm256_setzero_ps();
        sum_v[1] = _mm256_setzero_ps();
        sum_v[2] = _mm256_setzero_ps();

        sum_v[0] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp1),
            _mm256_loadu_ps(input_temp1), sum_v[0]);

        sum_v[0] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp1 + 8),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp1 + 8),
                _mm256_loadu_ps(input_temp1 + WIDTH - 3),
                0xf8), // 1111 1000
            sum_v[0]);

        sum_v[0] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp1 + 16),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp1 + WIDTH + 5),
                _mm256_loadu_ps(input_temp1 + 2*WIDTH - 6),
                0xc0), // 1100 0000
            sum_v[0]);

        sum_v[0] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp1 + 24),
            _mm256_loadu_ps(input_temp1 + 2*WIDTH + 2),
            sum_v[0]);

        sum_v[0] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp1 + 32),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp1 + 2*WIDTH + 10),
                _mm256_loadu_ps(input_temp1 + 3*WIDTH - 1),
                0xfe), // 1111 1110
            sum_v[0]);

        sum_v[0] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp1 + 40),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp1 + 3*WIDTH + 7),
                _mm256_loadu_ps(input_temp1 + 4*WIDTH - 4),
                0xf8), // 1111 0000
            sum_v[0]);

        sum_v[0] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp1 + 48),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp1 + 4*WIDTH + 4),
                _mm256_loadu_ps(input_temp1 + 5*WIDTH - 7),
                0x80), // 1000 0000
            sum_v[0]);

        sum_v[0] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp1 + 56),
            _mm256_loadu_ps(input_temp1 + 5*WIDTH + 1),
            sum_v[0]);

        sum_v[0] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp1 + 64),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp1 + 5*WIDTH + 9),
                _mm256_loadu_ps(input_temp1 + 6*WIDTH - 2),
                0xfc), // 1111 1100
            sum_v[0]);

        sum_v[0] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp1 + 72),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp1 + 6*WIDTH + 6),
                _mm256_loadu_ps(input_temp1 + 7*WIDTH - 5),
                0xe0), // 1110 0000
            sum_v[0]);

        sum_v[0] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp1 + 80),
            _mm256_loadu_ps(input_temp1 + 7*WIDTH + 3),
            sum_v[0]);

        sum_v[0] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp1 + 88),
            _mm256_loadu_ps(input_temp1 + 8*WIDTH),
            sum_v[0]);

        sum_v[0] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp1 + 96),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp1 + 8*WIDTH + 8),
                _mm256_loadu_ps(input_temp1 + 9*WIDTH - 3),
                0xf8), // 1111 1000
            sum_v[0]);

        sum_v[0] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp1 + 104),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp1 + 9*WIDTH + 5),
                _mm256_loadu_ps(input_temp1 + 10*WIDTH - 6),
                0xc0), // 1100 0000
            sum_v[0]);

        sum_v[0] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp1 + 112),
            _mm256_loadu_ps(input_temp1 + 10*WIDTH + 2),
            sum_v[0]);

        sum_v[0] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp1 + 120),
            _mm256_maskload_ps(input_temp1 + 10*WIDTH + 10, mask1),
            sum_v[0]);

        _mm_prefetch((const char *)(input_temp2 + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 2*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 3*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 4*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 5*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 6*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 7*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 8*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 9*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 10*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);

        sum_v[1] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp2),
            _mm256_loadu_ps(input_temp2), sum_v[1]);

        sum_v[1] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp2 + 8),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp2 + 8),
                _mm256_loadu_ps(input_temp2 + WIDTH - 3),
                0xf8), // 1111 1000
            sum_v[1]);

        sum_v[1] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp2 + 16),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp2 + WIDTH + 5),
                _mm256_loadu_ps(input_temp2 + 2*WIDTH - 6),
                0xc0), // 1100 0000
            sum_v[1]);

        sum_v[1] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp2 + 24),
            _mm256_loadu_ps(input_temp2 + 2*WIDTH + 2),
            sum_v[1]);

        sum_v[1] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp2 + 32),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp2 + 2*WIDTH + 10),
                _mm256_loadu_ps(input_temp2 + 3*WIDTH - 1),
                0xfe), // 1111 1110
            sum_v[1]);

        sum_v[1] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp2 + 40),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp2 + 3*WIDTH + 7),
                _mm256_loadu_ps(input_temp2 + 4*WIDTH - 4),
                0xf8), // 1111 0000
            sum_v[1]);

        sum_v[1] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp2 + 48),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp2 + 4*WIDTH + 4),
                _mm256_loadu_ps(input_temp2 + 5*WIDTH - 7),
                0x80), // 1000 0000
            sum_v[1]);

        sum_v[1] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp2 + 56),
            _mm256_loadu_ps(input_temp2 + 5*WIDTH + 1),
            sum_v[1]);

        sum_v[1] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp2 + 64),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp2 + 5*WIDTH + 9),
                _mm256_loadu_ps(input_temp2 + 6*WIDTH - 2),
                0xfc), // 1111 1100
            sum_v[1]);

        sum_v[1] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp2 + 72),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp2 + 6*WIDTH + 6),
                _mm256_loadu_ps(input_temp2 + 7*WIDTH - 5),
                0xe0), // 1110 0000
            sum_v[1]);

        sum_v[1] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp2 + 80),
            _mm256_loadu_ps(input_temp2 + 7*WIDTH + 3),
            sum_v[1]);

        sum_v[1] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp2 + 88),
            _mm256_loadu_ps(input_temp2 + 8*WIDTH),
            sum_v[1]);

        sum_v[1] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp2 + 96),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp2 + 8*WIDTH + 8),
                _mm256_loadu_ps(input_temp2 + 9*WIDTH - 3),
                0xf8), // 1111 1000
            sum_v[1]);

        sum_v[1] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp2 + 104),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp2 + 9*WIDTH + 5),
                _mm256_loadu_ps(input_temp2 + 10*WIDTH - 6),
                0xc0), // 1100 0000
            sum_v[1]);

        sum_v[1] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp2 + 112),
            _mm256_loadu_ps(input_temp2 + 10*WIDTH + 2),
            sum_v[1]);

        sum_v[1] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp2 + 120),
            _mm256_maskload_ps(input_temp2 + 10*WIDTH + 10, mask1),
            sum_v[1]);

        _mm_prefetch((const char *)(input_temp3 + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp3 + WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp3 + 2*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp3 + 3*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp3 + 4*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp3 + 5*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp3 + 6*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp3 + 7*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp3 + 8*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp3 + 9*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp3 + 10*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);

        sum_v[2] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp3),
            _mm256_loadu_ps(input_temp3), sum_v[2]);

        sum_v[2] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp3 + 8),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp3 + 8),
                _mm256_loadu_ps(input_temp3 + WIDTH - 3),
                0xf8), // 1111 1000
            sum_v[2]);

        sum_v[2] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp3 + 16),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp3 + WIDTH + 5),
                _mm256_loadu_ps(input_temp3 + 2*WIDTH - 6),
                0xc0), // 1100 0000
            sum_v[2]);

        sum_v[2] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp3 + 24),
            _mm256_loadu_ps(input_temp3 + 2*WIDTH + 2),
            sum_v[2]);

        sum_v[2] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp3 + 32),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp3 + 2*WIDTH + 10),
                _mm256_loadu_ps(input_temp3 + 3*WIDTH - 1),
                0xfe), // 1111 1110
            sum_v[2]);

        sum_v[2] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp3 + 40),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp3 + 3*WIDTH + 7),
                _mm256_loadu_ps(input_temp3 + 4*WIDTH - 4),
                0xf8), // 1111 0000
            sum_v[2]);

        sum_v[2] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp3 + 48),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp3 + 4*WIDTH + 4),
                _mm256_loadu_ps(input_temp3 + 5*WIDTH - 7),
                0x80), // 1000 0000
            sum_v[2]);

        sum_v[2] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp3 + 56),
            _mm256_loadu_ps(input_temp3 + 5*WIDTH + 1),
            sum_v[2]);

        sum_v[2] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp3 + 64),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp3 + 5*WIDTH + 9),
                _mm256_loadu_ps(input_temp3 + 6*WIDTH - 2),
                0xfc), // 1111 1100
            sum_v[2]);

        sum_v[2] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp3 + 72),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp3 + 6*WIDTH + 6),
                _mm256_loadu_ps(input_temp3 + 7*WIDTH - 5),
                0xe0), // 1110 0000
            sum_v[2]);

        sum_v[2] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp3 + 80),
            _mm256_loadu_ps(input_temp3 + 7*WIDTH + 3),
            sum_v[2]);

        sum_v[2] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp3 + 88),
            _mm256_loadu_ps(input_temp3 + 8*WIDTH),
            sum_v[2]);

        sum_v[2] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp3 + 96),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp3 + 8*WIDTH + 8),
                _mm256_loadu_ps(input_temp3 + 9*WIDTH - 3),
                0xf8), // 1111 1000
            sum_v[2]);

        sum_v[2] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp3 + 104),
            _mm256_blend_ps(
                _mm256_loadu_ps(input_temp3 + 9*WIDTH + 5),
                _mm256_loadu_ps(input_temp3 + 10*WIDTH - 6),
                0xc0), // 1100 0000
            sum_v[2]);

        sum_v[2] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp3 + 112),
            _mm256_loadu_ps(input_temp3 + 10*WIDTH + 2),
            sum_v[2]);

        sum_v[2] = _mm256_fmadd_ps(
            _mm256_load_ps(weight_temp3 + 120),
            _mm256_maskload_ps(input_temp3 + 10*WIDTH + 10, mask1),
            sum_v[2]);

        _mm256_store_ps(sum_temp, _mm256_add_ps(_mm256_add_ps(sum_v[0], sum_v[1]), sum_v[2]));
        float sum = 0;
        for (int i = 0; i < 8; ++i) {
          sum += sum_temp[i];
        }

        output[(out_channel * WOUT + output_row) * WOUT + output_col] = sum;
      } // for each output col
    } // for each output row
  } // for each out channel
}

// output channel unrolled twice
// input channel fully unrolled
inline void conv1_ver2(const float *weight, const float *input, float *output)
{
  const int M = 96;
  const int WIDTH = 227;
  const int STRIDE = 4;
  const int K = 11;
  const int WOUT = (WIDTH - K)/STRIDE + 1; // 55

  const int WOUT_BLOCK = 55;
  const int KERNEL_SIZE_ALIGNED = 128;

  __declspec(aligned(64)) float sum_temp[8];
  int mask1_temp[8] = { -1, 0, 0, 0, 0, 0, 0, 0 };
  __m256i mask1 = _mm256_load_si256((__m256i *)mask1_temp);

  for (int out_channel = 0; out_channel < M; out_channel += 2) {
    for (int output_row = 0; output_row < WOUT; ++output_row) {
      for (int output_col = 0; output_col < WOUT; ++output_col) {
        const float *input_temp0 = input + (output_row * WIDTH + output_col) * STRIDE;
        const float *input_temp1 = input_temp0 + WIDTH * WIDTH;
        const float *input_temp2 = input_temp0 + 2 * WIDTH * WIDTH;

        const float *weight_temp0 = weight + 3 * out_channel * KERNEL_SIZE_ALIGNED;
        const float *weight_temp1 = weight_temp0 + KERNEL_SIZE_ALIGNED;
        const float *weight_temp2 = weight_temp0 + 2 * KERNEL_SIZE_ALIGNED;

        const float *weight_temp3 = weight + 3 * (out_channel + 1) * KERNEL_SIZE_ALIGNED;
        const float *weight_temp4 = weight_temp3 + KERNEL_SIZE_ALIGNED;
        const float *weight_temp5 = weight_temp3 + 2 * KERNEL_SIZE_ALIGNED;

        const int PREFETCH_DISTANCE = 16;

        _mm_prefetch((const char *)(input_temp0 + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 2*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 3*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 4*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 5*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 6*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 7*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 8*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 9*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 10*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);

        __m256 sum_v[3];
        sum_v[0] = _mm256_setzero_ps();
        sum_v[1] = _mm256_setzero_ps();
        sum_v[2] = _mm256_setzero_ps();

        sum_v[3] = _mm256_setzero_ps();
        sum_v[4] = _mm256_setzero_ps();
        sum_v[5] = _mm256_setzero_ps();

        __m256 in_v;

        in_v = _mm256_loadu_ps(input_temp0);

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3), in_v, sum_v[3]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 8),
            _mm256_loadu_ps(input_temp0 + WIDTH - 3),
            0xf8); // 1111 1000

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 8), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 8), in_v, sum_v[3]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + WIDTH + 5),
            _mm256_loadu_ps(input_temp0 + 2*WIDTH - 6),
            0xc0); // 1100 0000

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 16), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 16), in_v, sum_v[3]);

        in_v = _mm256_loadu_ps(input_temp0 + 2*WIDTH + 2);

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 24), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 24), in_v, sum_v[3]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 2*WIDTH + 10),
            _mm256_loadu_ps(input_temp0 + 3*WIDTH - 1),
            0xfe); // 1111 1110

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 32), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 32), in_v, sum_v[3]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 3*WIDTH + 7),
            _mm256_loadu_ps(input_temp0 + 4*WIDTH - 4),
            0xf8); // 1111 0000

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 40), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 40), in_v, sum_v[3]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 4*WIDTH + 4),
            _mm256_loadu_ps(input_temp0 + 5*WIDTH - 7),
            0x80); // 1000 0000

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 48), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 48), in_v, sum_v[3]);

        in_v = _mm256_loadu_ps(input_temp0 + 5*WIDTH + 1);

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 56), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 56), in_v, sum_v[3]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 5*WIDTH + 9),
            _mm256_loadu_ps(input_temp0 + 6*WIDTH - 2),
            0xfc); // 1111 1100

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 64), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 64), in_v, sum_v[3]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 6*WIDTH + 6),
            _mm256_loadu_ps(input_temp0 + 7*WIDTH - 5),
            0xe0); // 1110 0000

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 72), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 72), in_v, sum_v[3]);

        in_v = _mm256_loadu_ps(input_temp0 + 7*WIDTH + 3);

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 80), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 80), in_v, sum_v[3]);

        in_v = _mm256_loadu_ps(input_temp0 + 8*WIDTH);

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 88), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 88), in_v, sum_v[3]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 8*WIDTH + 8),
            _mm256_loadu_ps(input_temp0 + 9*WIDTH - 3),
            0xf8); // 1111 1000

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 96), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 96), in_v, sum_v[3]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 9*WIDTH + 5),
            _mm256_loadu_ps(input_temp0 + 10*WIDTH - 6),
            0xc0); // 1100 0000

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 104), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 104), in_v, sum_v[3]);

        in_v = _mm256_loadu_ps(input_temp0 + 10*WIDTH + 2);

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 112), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 112), in_v, sum_v[3]);

        in_v = _mm256_maskload_ps(input_temp0 + 10*WIDTH + 10, mask1);

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 120), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 120), in_v, sum_v[3]);

        _mm_prefetch((const char *)(input_temp1 + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 2*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 3*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 4*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 5*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 6*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 7*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 8*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 9*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 10*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);

        in_v = _mm256_loadu_ps(input_temp1);

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4), in_v, sum_v[4]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 8),
            _mm256_loadu_ps(input_temp1 + WIDTH - 3),
            0xf8); // 1111 1000

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 8), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 8), in_v, sum_v[4]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + WIDTH + 5),
            _mm256_loadu_ps(input_temp1 + 2*WIDTH - 6),
            0xc0); // 1100 0000

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 16), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 16), in_v, sum_v[4]);

        in_v = _mm256_loadu_ps(input_temp1 + 2*WIDTH + 2);

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 24), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 24), in_v, sum_v[4]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 2*WIDTH + 10),
            _mm256_loadu_ps(input_temp1 + 3*WIDTH - 1),
            0xfe); // 1111 1110

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 32), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 32), in_v, sum_v[4]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 3*WIDTH + 7),
            _mm256_loadu_ps(input_temp1 + 4*WIDTH - 4),
            0xf8); // 1111 0000

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 40), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 40), in_v, sum_v[4]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 4*WIDTH + 4),
            _mm256_loadu_ps(input_temp1 + 5*WIDTH - 7),
            0x80); // 1000 0000

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 48), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 48), in_v, sum_v[4]);

        in_v = _mm256_loadu_ps(input_temp1 + 5*WIDTH + 1);

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 56), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 56), in_v, sum_v[4]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 5*WIDTH + 9),
            _mm256_loadu_ps(input_temp1 + 6*WIDTH - 2),
            0xfc); // 1111 1100

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 64), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 64), in_v, sum_v[4]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 6*WIDTH + 6),
            _mm256_loadu_ps(input_temp1 + 7*WIDTH - 5),
            0xe0); // 1110 0000

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 72), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 72), in_v, sum_v[4]);

        in_v = _mm256_loadu_ps(input_temp1 + 7*WIDTH + 3);

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 80), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 80), in_v, sum_v[4]);

        in_v = _mm256_loadu_ps(input_temp1 + 8*WIDTH);

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 88), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 88), in_v, sum_v[4]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 8*WIDTH + 8),
            _mm256_loadu_ps(input_temp1 + 9*WIDTH - 3),
            0xf8); // 1111 1000

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 96), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 96), in_v, sum_v[4]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 9*WIDTH + 5),
            _mm256_loadu_ps(input_temp1 + 10*WIDTH - 6),
            0xc0); // 1100 0000

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 104), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 104), in_v, sum_v[4]);

        in_v = _mm256_loadu_ps(input_temp1 + 10*WIDTH + 2);

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 112), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 112), in_v, sum_v[4]);

        in_v = _mm256_maskload_ps(input_temp1 + 10*WIDTH + 10, mask1);

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 120), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 120), in_v, sum_v[4]);

        _mm_prefetch((const char *)(input_temp2 + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 2*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 3*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 4*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 5*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 6*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 7*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 8*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 9*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 10*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);

        in_v = _mm256_loadu_ps(input_temp2);

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5), in_v, sum_v[5]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 8),
            _mm256_loadu_ps(input_temp2 + WIDTH - 3),
            0xf8); // 1111 1000

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 8), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 8), in_v, sum_v[5]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + WIDTH + 5),
            _mm256_loadu_ps(input_temp2 + 2*WIDTH - 6),
            0xc0); // 1100 0000

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 16), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 16), in_v, sum_v[5]);

        in_v = _mm256_loadu_ps(input_temp2 + 2*WIDTH + 2);

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 24), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 24), in_v, sum_v[5]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 2*WIDTH + 10),
            _mm256_loadu_ps(input_temp2 + 3*WIDTH - 1),
            0xfe); // 1111 1110

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 32), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 32), in_v, sum_v[5]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 3*WIDTH + 7),
            _mm256_loadu_ps(input_temp2 + 4*WIDTH - 4),
            0xf8); // 1111 0000

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 40), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 40), in_v, sum_v[5]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 4*WIDTH + 4),
            _mm256_loadu_ps(input_temp2 + 5*WIDTH - 7),
            0x80); // 1000 0000

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 48), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 48), in_v, sum_v[5]);

        in_v = _mm256_loadu_ps(input_temp2 + 5*WIDTH + 1);

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 56), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 56), in_v, sum_v[5]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 5*WIDTH + 9),
            _mm256_loadu_ps(input_temp2 + 6*WIDTH - 2),
            0xfc); // 1111 1100

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 64), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 64), in_v, sum_v[5]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 6*WIDTH + 6),
            _mm256_loadu_ps(input_temp2 + 7*WIDTH - 5),
            0xe0); // 1110 0000

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 72), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 72), in_v, sum_v[5]);

        in_v = _mm256_loadu_ps(input_temp2 + 7*WIDTH + 3);

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 80), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 80), in_v, sum_v[5]);

        in_v = _mm256_loadu_ps(input_temp2 + 8*WIDTH);

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 88), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 88), in_v, sum_v[5]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 8*WIDTH + 8),
            _mm256_loadu_ps(input_temp2 + 9*WIDTH - 3),
            0xf8); // 1111 1000

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 96), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 96), in_v, sum_v[5]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 9*WIDTH + 5),
            _mm256_loadu_ps(input_temp2 + 10*WIDTH - 6),
            0xc0); // 1100 0000

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 104), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 104), in_v, sum_v[5]);

        in_v = _mm256_loadu_ps(input_temp2 + 10*WIDTH + 2);

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 112), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 112), in_v, sum_v[5]);

        in_v = _mm256_maskload_ps(input_temp2 + 10*WIDTH + 10, mask1);

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 120), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 120), in_v, sum_v[5]);

        _mm256_store_ps(sum_temp, _mm256_add_ps(_mm256_add_ps(sum_v[0], sum_v[1]), sum_v[2]));
        float sum = 0;
        for (int i = 0; i < 8; ++i) {
          sum += sum_temp[i];
        }

        output[(out_channel * WOUT + output_row) * WOUT + output_col] = sum;

        _mm256_store_ps(sum_temp, _mm256_add_ps(_mm256_add_ps(sum_v[3], sum_v[4]), sum_v[5]));
        sum = 0;
        for (int i = 0; i < 8; ++i) {
          sum += sum_temp[i];
        }

        output[((out_channel + 1) * WOUT + output_row) * WOUT + output_col] = sum;
      } // for each output col
    } // for each output row
  } // for each out channel
}

// output channel unrolled twice
inline void conv1_ver6(const float *weight, const float *input, float *output)
{
  const int M = 96;
  const int WIDTH = 227;
  const int STRIDE = 4;
  const int K = 11;
  const int WOUT = (WIDTH - K)/STRIDE + 1; // 55

  const int WOUT_BLOCK = 55;
//  const int KERNEL_SIZE_ALIGNED = 128;
  const int KERNEL_SIZE_ALIGNED = 176;

  __declspec(aligned(64)) float sum_temp0[8], sum_temp1[8], sum_temp2[8], sum_temp3[8];
  int mask1_temp[8] = { -1, 0, 0, 0, 0, 0, 0, 0 };
  __m256i mask1 = _mm256_load_si256((__m256i *)mask1_temp);

  for (int out_channel = 0; out_channel < M; ++out_channel) {
    for (int output_row = 0; output_row < WOUT; ++output_row) {
      for (int output_col = 0; output_col < 48; output_col += 4) {
        __m256 sum_v[4];
        sum_v[0] = _mm256_setzero_ps();
        sum_v[1] = _mm256_setzero_ps();
        sum_v[2] = _mm256_setzero_ps();
        sum_v[3] = _mm256_setzero_ps();

        for (int in_channel = 0; in_channel < 3; ++in_channel) {
          const float *input_temp = input + (output_row * WIDTH + output_col) * STRIDE + in_channel * WIDTH * WIDTH;
          const float *weight_temp = weight + (3 * out_channel + in_channel) * KERNEL_SIZE_ALIGNED;

          const int PREFETCH_DISTANCE = 16;

#define MY_FMADD(K) \
          in0_v = _mm256_loadu_ps(input_temp + K*WIDTH); \
          in1_v = _mm256_loadu_ps(input_temp + K*WIDTH + 4); \
          in2_v = _mm256_loadu_ps(input_temp + K*WIDTH + 8); \
          in3_v = _mm256_loadu_ps(input_temp + K*WIDTH + 12); \
          in4_v = _mm256_loadu_ps(input_temp + K*WIDTH + 16); \
          in5_v = _mm256_loadu_ps(input_temp + K*WIDTH + 20); \
          w_v = _mm256_load_ps(weight_temp + K*16); \
          sum_v[0] = _mm256_fmadd_ps(w_v, in0_v, sum_v[0]); \
          sum_v[1] = _mm256_fmadd_ps(w_v, in1_v, sum_v[1]); \
          sum_v[2] = _mm256_fmadd_ps(w_v, in2_v, sum_v[2]); \
          sum_v[3] = _mm256_fmadd_ps(w_v, in3_v, sum_v[3]); \
          w_v = _mm256_load_ps(weight_temp + K*16 + 8); \
          sum_v[0] = _mm256_fmadd_ps(w_v, in2_v, sum_v[0]); \
          sum_v[1] = _mm256_fmadd_ps(w_v, in3_v, sum_v[1]); \
          sum_v[2] = _mm256_fmadd_ps(w_v, in4_v, sum_v[2]); \
          sum_v[3] = _mm256_fmadd_ps(w_v, in5_v, sum_v[3]);

          _mm_prefetch((const char *)(input_temp + PREFETCH_DISTANCE), _MM_HINT_T0);
          _mm_prefetch((const char *)(input_temp + WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
          _mm_prefetch((const char *)(input_temp + 2*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
          _mm_prefetch((const char *)(input_temp + 3*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
          _mm_prefetch((const char *)(input_temp + 4*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
          _mm_prefetch((const char *)(input_temp + 5*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
          _mm_prefetch((const char *)(input_temp + 6*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
          _mm_prefetch((const char *)(input_temp + 7*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
          _mm_prefetch((const char *)(input_temp + 8*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
          _mm_prefetch((const char *)(input_temp + 9*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
          _mm_prefetch((const char *)(input_temp + 10*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);

          __m256 in0_v, in1_v, in2_v, in3_v, in4_v, in5_v;
          __m256 w_v;

          int off = 0;
          MY_FMADD(0);
          MY_FMADD(1);
          MY_FMADD(2);
          MY_FMADD(3);
          MY_FMADD(4);
          MY_FMADD(5);
          MY_FMADD(6);
          MY_FMADD(7);
          MY_FMADD(8);
          MY_FMADD(9);
          MY_FMADD(10);

        } // for each input channel

        float sum;

        _mm256_store_ps(sum_temp0, sum_v[0]);
        sum = 0;
        for (int i = 0; i < 8; ++i) {
          sum += sum_temp0[i];
        }
        output[(out_channel * WOUT + output_row) * WOUT + output_col] = sum;

        _mm256_store_ps(sum_temp1, sum_v[1]);
        sum = 0;
        for (int i = 0; i < 8; ++i) {
          sum += sum_temp1[i];
        }
        output[(out_channel * WOUT + output_row) * WOUT + output_col + 1] = sum;

        _mm256_store_ps(sum_temp2, sum_v[2]);
        sum = 0;
        for (int i = 0; i < 8; ++i) {
          sum += sum_temp2[i];
        }
        output[(out_channel * WOUT + output_row) * WOUT + output_col + 2] = sum;

        _mm256_store_ps(sum_temp3, sum_v[3]);
        sum = 0;
        for (int i = 0; i < 8; ++i) {
          sum += sum_temp3[i];
        }
        output[(out_channel * WOUT + output_row) * WOUT + output_col + 3] = sum;
      } // for each output col

      for (int output_col = 0; output_col < WOUT; ++output_col) {
        __m256 sum_v[1];
        sum_v[0] = _mm256_setzero_ps();

        for (int in_channel = 0; in_channel < 3; ++in_channel) {
          const float *input_temp = input + (output_row * WIDTH + output_col) * STRIDE + in_channel * WIDTH * WIDTH;

          const float *weight_temp = weight + (3 * out_channel + in_channel) * KERNEL_SIZE_ALIGNED;

          const int PREFETCH_DISTANCE = 16;

          _mm_prefetch((const char *)(input_temp + PREFETCH_DISTANCE), _MM_HINT_T0);
          _mm_prefetch((const char *)(input_temp + WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
          _mm_prefetch((const char *)(input_temp + 2*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
          _mm_prefetch((const char *)(input_temp + 3*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
          _mm_prefetch((const char *)(input_temp + 4*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
          _mm_prefetch((const char *)(input_temp + 5*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
          _mm_prefetch((const char *)(input_temp + 6*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
          _mm_prefetch((const char *)(input_temp + 7*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
          _mm_prefetch((const char *)(input_temp + 8*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
          _mm_prefetch((const char *)(input_temp + 9*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
          _mm_prefetch((const char *)(input_temp + 10*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);

          __m256 in_v;

          in_v = _mm256_loadu_ps(input_temp);
          sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp), in_v, sum_v[0]);
          in_v = _mm256_loadu_ps(input_temp + 8);
          sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp + 8), in_v, sum_v[0]);

          in_v = _mm256_loadu_ps(input_temp + WIDTH);
          sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp + 16), in_v, sum_v[0]);
          in_v = _mm256_loadu_ps(input_temp + WIDTH + 8);
          sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp + 24), in_v, sum_v[0]);

          in_v = _mm256_loadu_ps(input_temp + 2*WIDTH);
          sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp + 32), in_v, sum_v[0]);
          in_v = _mm256_loadu_ps(input_temp + 2*WIDTH + 8);
          sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp + 40), in_v, sum_v[0]);

          in_v = _mm256_loadu_ps(input_temp + 3*WIDTH);
          sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp + 48), in_v, sum_v[0]);
          in_v = _mm256_loadu_ps(input_temp + 3*WIDTH + 8);
          sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp + 56), in_v, sum_v[0]);

          in_v = _mm256_loadu_ps(input_temp + 4*WIDTH);
          sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp + 64), in_v, sum_v[0]);
          in_v = _mm256_loadu_ps(input_temp + 4*WIDTH + 8);
          sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp + 72), in_v, sum_v[0]);

          in_v = _mm256_loadu_ps(input_temp + 5*WIDTH);
          sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp + 80), in_v, sum_v[0]);
          in_v = _mm256_loadu_ps(input_temp + 5*WIDTH + 8);
          sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp + 88), in_v, sum_v[0]);

          in_v = _mm256_loadu_ps(input_temp + 6*WIDTH);
          sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp + 96), in_v, sum_v[0]);
          in_v = _mm256_loadu_ps(input_temp + 6*WIDTH + 8);
          sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp + 104), in_v, sum_v[0]);

          in_v = _mm256_loadu_ps(input_temp + 7*WIDTH);
          sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp + 112), in_v, sum_v[0]);
          in_v = _mm256_loadu_ps(input_temp + 7*WIDTH + 8);
          sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp + 120), in_v, sum_v[0]);

          in_v = _mm256_loadu_ps(input_temp + 8*WIDTH);
          sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp + 128), in_v, sum_v[0]);
          in_v = _mm256_loadu_ps(input_temp + 8*WIDTH + 8);
          sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp + 136), in_v, sum_v[0]);

          in_v = _mm256_loadu_ps(input_temp + 9*WIDTH);
          sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp + 144), in_v, sum_v[0]);
          in_v = _mm256_loadu_ps(input_temp + 9*WIDTH + 8);
          sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp + 152), in_v, sum_v[0]);

          in_v = _mm256_loadu_ps(input_temp + 10*WIDTH);
          sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp + 160), in_v, sum_v[0]);
          in_v = _mm256_loadu_ps(input_temp + 10*WIDTH + 8);
          sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp + 168), in_v, sum_v[0]);
        } // for each input channel

        _mm256_store_ps(sum_temp0, sum_v[0]);
        float sum = 0;
        for (int i = 0; i < 8; ++i) {
          sum += sum_temp0[i];
        }

        output[(out_channel * WOUT + output_row) * WOUT + output_col] = sum;
      } // for each output col
    } // for each output row
  } // for each out channel

#undef MY_FMADD
}

// output channel unrolled twice
// input channel fully unrolled
// output column unrolled twice
inline void conv1_ver3(const float *weight, const float *input, float *output)
{
  const int M = 96;
  const int WIDTH = 227;
  const int STRIDE = 4;
  const int K = 11;
  const int WOUT = (WIDTH - K)/STRIDE + 1; // 55

  // JSP: AlexNet conv1
  // Input: 3 x 227 x 227 => 201 KB per channel, 604 KB total
  // Output: 96 x 55 x 55 => 12 KB per channel, 1.1 MB total
  // Weight: 96 x 3 x 11 x 11 => 0.5 KB per channel pair, 1.5 KB per output channel, 45 KB per input channel, 136 KB total
  //         No matter what we do, there's no reuse on weight across different channels (only reuse is within a channel pair)
  // FLOPS: 2 x 96 x 3 x 55 x 55 x 11 x 11 = 211 MFLOPS
  //
  // Approach 1: stream output channel
  //   For each output channel, read inputs from LLC (604 KB total) and read weights from memory
  //   Total memory access: input 604 KB, output 1.1 MB, weight 136 KB
  //   Total LLC access: input 96 x 604 KB
  //
  // Approach 2: stream input channel
  //   For each input channel, read outputs from LLC (1.1 MB total) and read weights from memory
  //   Total memory access: input 604 KB, output 1.1 MB, weight 136 KB
  //   Total LLC access: output 2 x 1.1 MB
  //
  // Approach 3: blocking
  //   For each input channel, read 1/8 of outputs from L2 (1.1/8 MB total) and read weights from memory
  //   Total memory access: input 604 KB, output 1.1 MB, weight 136 KB
  //   Total LLC access: input 7 x 604 KB

  const int WOUT_BLOCK = 55;
  const int KERNEL_SIZE_ALIGNED = 128;

  __declspec(aligned(64)) float sum_temp[8];
  int mask1_temp[8] = { -1, 0, 0, 0, 0, 0, 0, 0 };
  __m256i mask1 = _mm256_load_si256((__m256i *)mask1_temp);

  for (int out_channel = 0; out_channel < M; out_channel += 2) {
    for (int output_row = 0; output_row < WOUT; ++output_row) {
      for (int output_col = 0; output_col < WOUT - 1; output_col += 2) {
        const float *input_temp0 = input + (output_row * WIDTH + output_col) * STRIDE;
        const float *input_temp1 = input_temp0 + WIDTH * WIDTH;
        const float *input_temp2 = input_temp0 + 2 * WIDTH * WIDTH;

        const float *weight_temp0 = weight + 3 * out_channel * KERNEL_SIZE_ALIGNED;
        const float *weight_temp1 = weight_temp0 + KERNEL_SIZE_ALIGNED;
        const float *weight_temp2 = weight_temp0 + 2 * KERNEL_SIZE_ALIGNED;

        const float *weight_temp3 = weight + 3 * (out_channel + 1) * KERNEL_SIZE_ALIGNED;
        const float *weight_temp4 = weight_temp3 + KERNEL_SIZE_ALIGNED;
        const float *weight_temp5 = weight_temp3 + 2 * KERNEL_SIZE_ALIGNED;

        const int PREFETCH_DISTANCE = 16;

        _mm_prefetch((const char *)(input_temp0 + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 2*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 3*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 4*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 5*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 6*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 7*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 8*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 9*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 10*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);

        __m256 sum_v[12];
        sum_v[0] = _mm256_setzero_ps();
        sum_v[1] = _mm256_setzero_ps();
        sum_v[2] = _mm256_setzero_ps();

        sum_v[3] = _mm256_setzero_ps();
        sum_v[4] = _mm256_setzero_ps();
        sum_v[5] = _mm256_setzero_ps();

        sum_v[6] = _mm256_setzero_ps();
        sum_v[7] = _mm256_setzero_ps();
        sum_v[8] = _mm256_setzero_ps();

        sum_v[9] = _mm256_setzero_ps();
        sum_v[10] = _mm256_setzero_ps();
        sum_v[11] = _mm256_setzero_ps();

        __m256 in0_v, in1_v;
        __m256 w0_v, w1_v;

#define MY_FMADD \
        sum_v[0] = _mm256_fmadd_ps(w0_v, in0_v, sum_v[0]); \
        sum_v[3] = _mm256_fmadd_ps(w1_v, in0_v, sum_v[3]); \
        sum_v[6] = _mm256_fmadd_ps(w0_v, in1_v, sum_v[6]); \
        sum_v[9] = _mm256_fmadd_ps(w1_v, in1_v, sum_v[9]);

        in0_v = _mm256_loadu_ps(input_temp0);
        in1_v = _mm256_loadu_ps(input_temp0 + 4);

        w0_v = _mm256_load_ps(weight_temp0);
        w1_v = _mm256_load_ps(weight_temp3);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 8),
            _mm256_loadu_ps(input_temp0 + WIDTH - 3),
            0xf8); // 1111 1000
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 12),
            _mm256_loadu_ps(input_temp0 + WIDTH + 1),
            0xf8); // 1111 1000

        w0_v = _mm256_load_ps(weight_temp0 + 8);
        w1_v = _mm256_load_ps(weight_temp3 + 8);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + WIDTH + 5),
            _mm256_loadu_ps(input_temp0 + 2*WIDTH - 6),
            0xc0); // 1100 0000
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + WIDTH + 9),
            _mm256_loadu_ps(input_temp0 + 2*WIDTH - 2),
            0xc0); // 1100 0000

        w0_v = _mm256_load_ps(weight_temp0 + 16);
        w1_v = _mm256_load_ps(weight_temp3 + 16);

        MY_FMADD

        in0_v = _mm256_loadu_ps(input_temp0 + 2*WIDTH + 2);
        in1_v = _mm256_loadu_ps(input_temp0 + 2*WIDTH + 6);

        w0_v = _mm256_load_ps(weight_temp0 + 24);
        w1_v = _mm256_load_ps(weight_temp3 + 24);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 2*WIDTH + 10),
            _mm256_loadu_ps(input_temp0 + 3*WIDTH - 1),
            0xfe); // 1111 1110
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 2*WIDTH + 14),
            _mm256_loadu_ps(input_temp0 + 3*WIDTH + 3),
            0xfe); // 1111 1110

        w0_v = _mm256_load_ps(weight_temp0 + 32);
        w1_v = _mm256_load_ps(weight_temp3 + 32);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 3*WIDTH + 7),
            _mm256_loadu_ps(input_temp0 + 4*WIDTH - 4),
            0xf8); // 1111 0000
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 3*WIDTH + 11),
            _mm256_loadu_ps(input_temp0 + 4*WIDTH),
            0xf8); // 1111 0000

        w0_v = _mm256_load_ps(weight_temp0 + 40);
        w1_v = _mm256_load_ps(weight_temp3 + 40);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 4*WIDTH + 4),
            _mm256_loadu_ps(input_temp0 + 5*WIDTH - 7),
            0x80); // 1000 0000
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 4*WIDTH + 8),
            _mm256_loadu_ps(input_temp0 + 5*WIDTH - 3),
            0x80); // 1000 0000

        w0_v = _mm256_load_ps(weight_temp0 + 48);
        w1_v = _mm256_load_ps(weight_temp3 + 48);

        MY_FMADD

        in0_v = _mm256_loadu_ps(input_temp0 + 5*WIDTH + 1);
        in1_v = _mm256_loadu_ps(input_temp0 + 5*WIDTH + 5);

        w0_v = _mm256_load_ps(weight_temp0 + 56);
        w1_v = _mm256_load_ps(weight_temp3 + 56);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 5*WIDTH + 9),
            _mm256_loadu_ps(input_temp0 + 6*WIDTH - 2),
            0xfc); // 1111 1100
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 5*WIDTH + 13),
            _mm256_loadu_ps(input_temp0 + 6*WIDTH + 2),
            0xfc); // 1111 1100

        w0_v = _mm256_load_ps(weight_temp0 + 64);
        w1_v = _mm256_load_ps(weight_temp3 + 64);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 6*WIDTH + 6),
            _mm256_loadu_ps(input_temp0 + 7*WIDTH - 5),
            0xe0); // 1110 0000
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 6*WIDTH + 10),
            _mm256_loadu_ps(input_temp0 + 7*WIDTH - 1),
            0xe0); // 1110 0000

        w0_v = _mm256_load_ps(weight_temp0 + 72);
        w1_v = _mm256_load_ps(weight_temp3 + 72);

        MY_FMADD

        in0_v = _mm256_loadu_ps(input_temp0 + 7*WIDTH + 3);
        in1_v = _mm256_loadu_ps(input_temp0 + 7*WIDTH + 7);

        w0_v = _mm256_load_ps(weight_temp0 + 80);
        w1_v = _mm256_load_ps(weight_temp3 + 80);

        MY_FMADD

        in0_v = _mm256_loadu_ps(input_temp0 + 8*WIDTH);
        in1_v = _mm256_loadu_ps(input_temp0 + 8*WIDTH + 4);

        w0_v = _mm256_load_ps(weight_temp0 + 88);
        w1_v = _mm256_load_ps(weight_temp3 + 88);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 8*WIDTH + 8),
            _mm256_loadu_ps(input_temp0 + 9*WIDTH - 3),
            0xf8); // 1111 1000
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 8*WIDTH + 12),
            _mm256_loadu_ps(input_temp0 + 9*WIDTH + 1),
            0xf8); // 1111 1000

        w0_v = _mm256_load_ps(weight_temp0 + 96);
        w1_v = _mm256_load_ps(weight_temp3 + 96);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 9*WIDTH + 5),
            _mm256_loadu_ps(input_temp0 + 10*WIDTH - 6),
            0xc0); // 1100 0000
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 9*WIDTH + 9),
            _mm256_loadu_ps(input_temp0 + 10*WIDTH - 2),
            0xc0); // 1100 0000

        w0_v = _mm256_load_ps(weight_temp0 + 104);
        w1_v = _mm256_load_ps(weight_temp3 + 104);

        MY_FMADD

        in0_v = _mm256_loadu_ps(input_temp0 + 10*WIDTH + 2);
        in1_v = _mm256_loadu_ps(input_temp0 + 10*WIDTH + 6);

        w0_v = _mm256_load_ps(weight_temp0 + 112);
        w1_v = _mm256_load_ps(weight_temp3 + 112);

        MY_FMADD

        in0_v = _mm256_maskload_ps(input_temp0 + 10*WIDTH + 10, mask1);
        in1_v = _mm256_maskload_ps(input_temp0 + 10*WIDTH + 14, mask1);

        w0_v = _mm256_load_ps(weight_temp0 + 120);
        w1_v = _mm256_load_ps(weight_temp3 + 120);

        MY_FMADD

        _mm_prefetch((const char *)(input_temp1 + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 2*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 3*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 4*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 5*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 6*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 7*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 8*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 9*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 10*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);

#undef MY_FMADD
#define MY_FMADD \
        sum_v[1] = _mm256_fmadd_ps(w0_v, in0_v, sum_v[1]); \
        sum_v[4] = _mm256_fmadd_ps(w1_v, in0_v, sum_v[4]); \
        sum_v[7] = _mm256_fmadd_ps(w0_v, in1_v, sum_v[7]); \
        sum_v[10] = _mm256_fmadd_ps(w1_v, in1_v, sum_v[10]);

        in0_v = _mm256_loadu_ps(input_temp1);
        in1_v = _mm256_loadu_ps(input_temp1 + 4);

        w0_v = _mm256_load_ps(weight_temp1);
        w1_v = _mm256_load_ps(weight_temp4);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 8),
            _mm256_loadu_ps(input_temp1 + WIDTH - 3),
            0xf8); // 1111 1000
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 12),
            _mm256_loadu_ps(input_temp1 + WIDTH + 1),
            0xf8); // 1111 1000

        w0_v = _mm256_load_ps(weight_temp1 + 8);
        w1_v = _mm256_load_ps(weight_temp4 + 8);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + WIDTH + 5),
            _mm256_loadu_ps(input_temp1 + 2*WIDTH - 6),
            0xc0); // 1100 0000
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + WIDTH + 9),
            _mm256_loadu_ps(input_temp1 + 2*WIDTH - 2),
            0xc0); // 1100 0000

        w0_v = _mm256_load_ps(weight_temp1 + 16);
        w1_v = _mm256_load_ps(weight_temp4 + 16);

        MY_FMADD

        in0_v = _mm256_loadu_ps(input_temp1 + 2*WIDTH + 2);
        in1_v = _mm256_loadu_ps(input_temp1 + 2*WIDTH + 6);

        w0_v = _mm256_load_ps(weight_temp1 + 24);
        w1_v = _mm256_load_ps(weight_temp4 + 24);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 2*WIDTH + 10),
            _mm256_loadu_ps(input_temp1 + 3*WIDTH - 1),
            0xfe); // 1111 1110
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 2*WIDTH + 14),
            _mm256_loadu_ps(input_temp1 + 3*WIDTH + 3),
            0xfe); // 1111 1110

        w0_v = _mm256_load_ps(weight_temp1 + 32);
        w1_v = _mm256_load_ps(weight_temp4 + 32);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 3*WIDTH + 7),
            _mm256_loadu_ps(input_temp1 + 4*WIDTH - 4),
            0xf8); // 1111 0000
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 3*WIDTH + 11),
            _mm256_loadu_ps(input_temp1 + 4*WIDTH),
            0xf8); // 1111 0000

        w0_v = _mm256_load_ps(weight_temp1 + 40);
        w1_v = _mm256_load_ps(weight_temp4 + 40);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 4*WIDTH + 4),
            _mm256_loadu_ps(input_temp1 + 5*WIDTH - 7),
            0x80); // 1000 0000
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 4*WIDTH + 8),
            _mm256_loadu_ps(input_temp1 + 5*WIDTH - 3),
            0x80); // 1000 0000

        w0_v = _mm256_load_ps(weight_temp1 + 48);
        w1_v = _mm256_load_ps(weight_temp4 + 48);

        MY_FMADD

        in0_v = _mm256_loadu_ps(input_temp1 + 5*WIDTH + 1);
        in1_v = _mm256_loadu_ps(input_temp1 + 5*WIDTH + 5);

        w0_v = _mm256_load_ps(weight_temp1 + 56);
        w1_v = _mm256_load_ps(weight_temp4 + 56);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 5*WIDTH + 9),
            _mm256_loadu_ps(input_temp1 + 6*WIDTH - 2),
            0xfc); // 1111 1100
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 5*WIDTH + 13),
            _mm256_loadu_ps(input_temp1 + 6*WIDTH + 2),
            0xfc); // 1111 1100

        w0_v = _mm256_load_ps(weight_temp1 + 64);
        w1_v = _mm256_load_ps(weight_temp4 + 64);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 6*WIDTH + 6),
            _mm256_loadu_ps(input_temp1 + 7*WIDTH - 5),
            0xe0); // 1110 0000
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 6*WIDTH + 10),
            _mm256_loadu_ps(input_temp1 + 7*WIDTH - 1),
            0xe0); // 1110 0000

        w0_v = _mm256_load_ps(weight_temp1 + 72);
        w1_v = _mm256_load_ps(weight_temp4 + 72);

        MY_FMADD

        in0_v = _mm256_loadu_ps(input_temp1 + 7*WIDTH + 3);
        in1_v = _mm256_loadu_ps(input_temp1 + 7*WIDTH + 7);

        w0_v = _mm256_load_ps(weight_temp1 + 80);
        w1_v = _mm256_load_ps(weight_temp4 + 80);

        MY_FMADD

        in0_v = _mm256_loadu_ps(input_temp1 + 8*WIDTH);
        in1_v = _mm256_loadu_ps(input_temp1 + 8*WIDTH + 4);

        w0_v = _mm256_load_ps(weight_temp1 + 88);
        w1_v = _mm256_load_ps(weight_temp4 + 88);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 8*WIDTH + 8),
            _mm256_loadu_ps(input_temp1 + 9*WIDTH - 3),
            0xf8); // 1111 1000
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 8*WIDTH + 12),
            _mm256_loadu_ps(input_temp1 + 9*WIDTH + 1),
            0xf8); // 1111 1000

        w0_v = _mm256_load_ps(weight_temp1 + 96);
        w1_v = _mm256_load_ps(weight_temp4 + 96);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 9*WIDTH + 5),
            _mm256_loadu_ps(input_temp1 + 10*WIDTH - 6),
            0xc0); // 1100 0000
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 9*WIDTH + 9),
            _mm256_loadu_ps(input_temp1 + 10*WIDTH - 2),
            0xc0); // 1100 0000

        w0_v = _mm256_load_ps(weight_temp1 + 104);
        w1_v = _mm256_load_ps(weight_temp4 + 104);

        MY_FMADD

        in0_v = _mm256_loadu_ps(input_temp1 + 10*WIDTH + 2);
        in1_v = _mm256_loadu_ps(input_temp1 + 10*WIDTH + 6);

        w0_v = _mm256_load_ps(weight_temp1 + 112);
        w1_v = _mm256_load_ps(weight_temp4 + 112);

        MY_FMADD

        in0_v = _mm256_maskload_ps(input_temp1 + 10*WIDTH + 10, mask1);
        in1_v = _mm256_maskload_ps(input_temp1 + 10*WIDTH + 14, mask1);

        w0_v = _mm256_load_ps(weight_temp1 + 120);
        w1_v = _mm256_load_ps(weight_temp4 + 120);

        MY_FMADD

        _mm_prefetch((const char *)(input_temp2 + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 2*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 3*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 4*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 5*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 6*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 7*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 8*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 9*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 10*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);

#undef MY_FMADD
#define MY_FMADD \
        sum_v[2] = _mm256_fmadd_ps(w0_v, in0_v, sum_v[2]); \
        sum_v[5] = _mm256_fmadd_ps(w1_v, in0_v, sum_v[5]); \
        sum_v[8] = _mm256_fmadd_ps(w0_v, in1_v, sum_v[8]); \
        sum_v[11] = _mm256_fmadd_ps(w1_v, in1_v, sum_v[11]);

        in0_v = _mm256_loadu_ps(input_temp2);
        in1_v = _mm256_loadu_ps(input_temp2 + 4);

        w0_v = _mm256_load_ps(weight_temp2);
        w1_v = _mm256_load_ps(weight_temp5);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 8),
            _mm256_loadu_ps(input_temp2 + WIDTH - 3),
            0xf8); // 1111 1000
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 12),
            _mm256_loadu_ps(input_temp2 + WIDTH + 1),
            0xf8); // 1111 1000

        w0_v = _mm256_load_ps(weight_temp2 + 8);
        w1_v = _mm256_load_ps(weight_temp5 + 8);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + WIDTH + 5),
            _mm256_loadu_ps(input_temp2 + 2*WIDTH - 6),
            0xc0); // 1100 0000
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + WIDTH + 9),
            _mm256_loadu_ps(input_temp2 + 2*WIDTH - 2),
            0xc0); // 1100 0000

        w0_v = _mm256_load_ps(weight_temp2 + 16);
        w1_v = _mm256_load_ps(weight_temp5 + 16);

        MY_FMADD

        in0_v = _mm256_loadu_ps(input_temp2 + 2*WIDTH + 2);
        in1_v = _mm256_loadu_ps(input_temp2 + 2*WIDTH + 6);

        w0_v = _mm256_load_ps(weight_temp2 + 24);
        w1_v = _mm256_load_ps(weight_temp5 + 24);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 2*WIDTH + 10),
            _mm256_loadu_ps(input_temp2 + 3*WIDTH - 1),
            0xfe); // 1111 1110
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 2*WIDTH + 14),
            _mm256_loadu_ps(input_temp2 + 3*WIDTH + 3),
            0xfe); // 1111 1110

        w0_v = _mm256_load_ps(weight_temp2 + 32);
        w1_v = _mm256_load_ps(weight_temp5 + 32);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 3*WIDTH + 7),
            _mm256_loadu_ps(input_temp2 + 4*WIDTH - 4),
            0xf8); // 1111 0000
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 3*WIDTH + 11),
            _mm256_loadu_ps(input_temp2 + 4*WIDTH),
            0xf8); // 1111 0000

        w0_v = _mm256_load_ps(weight_temp2 + 40);
        w1_v = _mm256_load_ps(weight_temp5 + 40);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 4*WIDTH + 4),
            _mm256_loadu_ps(input_temp2 + 5*WIDTH - 7),
            0x80); // 1000 0000
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 4*WIDTH + 8),
            _mm256_loadu_ps(input_temp2 + 5*WIDTH - 3),
            0x80); // 1000 0000

        w0_v = _mm256_load_ps(weight_temp2 + 48);
        w1_v = _mm256_load_ps(weight_temp5 + 48);

        MY_FMADD

        in0_v = _mm256_loadu_ps(input_temp2 + 5*WIDTH + 1);
        in1_v = _mm256_loadu_ps(input_temp2 + 5*WIDTH + 5);

        w0_v = _mm256_load_ps(weight_temp2 + 56);
        w1_v = _mm256_load_ps(weight_temp5 + 56);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 5*WIDTH + 9),
            _mm256_loadu_ps(input_temp2 + 6*WIDTH - 2),
            0xfc); // 1111 1100
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 5*WIDTH + 13),
            _mm256_loadu_ps(input_temp2 + 6*WIDTH + 2),
            0xfc); // 1111 1100

        w0_v = _mm256_load_ps(weight_temp2 + 64);
        w1_v = _mm256_load_ps(weight_temp5 + 64);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 6*WIDTH + 6),
            _mm256_loadu_ps(input_temp2 + 7*WIDTH - 5),
            0xe0); // 1110 0000
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 6*WIDTH + 10),
            _mm256_loadu_ps(input_temp2 + 7*WIDTH - 1),
            0xe0); // 1110 0000

        w0_v = _mm256_load_ps(weight_temp2 + 72);
        w1_v = _mm256_load_ps(weight_temp5 + 72);

        MY_FMADD

        in0_v = _mm256_loadu_ps(input_temp2 + 7*WIDTH + 3);
        in1_v = _mm256_loadu_ps(input_temp2 + 7*WIDTH + 7);

        w0_v = _mm256_load_ps(weight_temp2 + 80);
        w1_v = _mm256_load_ps(weight_temp5 + 80);

        MY_FMADD

        in0_v = _mm256_loadu_ps(input_temp2 + 8*WIDTH);
        in1_v = _mm256_loadu_ps(input_temp2 + 8*WIDTH + 4);

        w0_v = _mm256_load_ps(weight_temp2 + 88);
        w1_v = _mm256_load_ps(weight_temp5 + 88);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 8*WIDTH + 8),
            _mm256_loadu_ps(input_temp2 + 9*WIDTH - 3),
            0xf8); // 1111 1000
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 8*WIDTH + 12),
            _mm256_loadu_ps(input_temp2 + 9*WIDTH + 1),
            0xf8); // 1111 1000

        w0_v = _mm256_load_ps(weight_temp2 + 96);
        w1_v = _mm256_load_ps(weight_temp5 + 96);

        MY_FMADD

        in0_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 9*WIDTH + 5),
            _mm256_loadu_ps(input_temp2 + 10*WIDTH - 6),
            0xc0); // 1100 0000
        in1_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 9*WIDTH + 9),
            _mm256_loadu_ps(input_temp2 + 10*WIDTH - 2),
            0xc0); // 1100 0000

        w0_v = _mm256_load_ps(weight_temp2 + 104);
        w1_v = _mm256_load_ps(weight_temp5 + 104);

        MY_FMADD

        in0_v = _mm256_loadu_ps(input_temp2 + 10*WIDTH + 2);
        in1_v = _mm256_loadu_ps(input_temp2 + 10*WIDTH + 6);

        w0_v = _mm256_load_ps(weight_temp2 + 112);
        w1_v = _mm256_load_ps(weight_temp5 + 112);

        MY_FMADD

        in0_v = _mm256_maskload_ps(input_temp2 + 10*WIDTH + 10, mask1);
        in1_v = _mm256_maskload_ps(input_temp2 + 10*WIDTH + 14, mask1);

        w0_v = _mm256_load_ps(weight_temp2 + 120);
        w1_v = _mm256_load_ps(weight_temp5 + 120);

        MY_FMADD
#undef MY_FMADD

        _mm256_store_ps(sum_temp, _mm256_add_ps(_mm256_add_ps(sum_v[0], sum_v[1]), sum_v[2]));
        float sum = 0;
        for (int i = 0; i < 8; ++i) {
          sum += sum_temp[i];
        }

        output[(out_channel * WOUT + output_row) * WOUT + output_col] = sum;

        _mm256_store_ps(sum_temp, _mm256_add_ps(_mm256_add_ps(sum_v[3], sum_v[4]), sum_v[5]));
        sum = 0;
        for (int i = 0; i < 8; ++i) {
          sum += sum_temp[i];
        }

        output[((out_channel + 1) * WOUT + output_row) * WOUT + output_col] = sum;

        _mm256_store_ps(sum_temp, _mm256_add_ps(_mm256_add_ps(sum_v[6], sum_v[7]), sum_v[8]));
        sum = 0;
        for (int i = 0; i < 8; ++i) {
          sum += sum_temp[i];
        }

        output[(out_channel * WOUT + output_row) * WOUT + output_col + 1] = sum;

        _mm256_store_ps(sum_temp, _mm256_add_ps(_mm256_add_ps(sum_v[9], sum_v[10]), sum_v[11]));
        sum = 0;
        for (int i = 0; i < 8; ++i) {
          sum += sum_temp[i];
        }

        output[((out_channel + 1) * WOUT + output_row) * WOUT + output_col + 1] = sum;
      } // for each output col

      for (int output_col = WOUT - 1; output_col < WOUT; ++output_col) {
        const float *input_temp0 = input + (output_row * WIDTH + output_col) * STRIDE;
        const float *input_temp1 = input_temp0 + WIDTH * WIDTH;
        const float *input_temp2 = input_temp0 + 2 * WIDTH * WIDTH;

        const float *weight_temp0 = weight + 3 * out_channel * K * K;
        const float *weight_temp1 = weight_temp0 + K * K;
        const float *weight_temp2 = weight_temp0 + 2 * K * K;

        const float *weight_temp3 = weight + 3 * (out_channel + 1) * K * K;
        const float *weight_temp4 = weight_temp3 + K * K;
        const float *weight_temp5 = weight_temp3 + 2 * K * K;

        const int PREFETCH_DISTANCE = 16;

        _mm_prefetch((const char *)(input_temp0 + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 2*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 3*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 4*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 5*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 6*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 7*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 8*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 9*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp0 + 10*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);

        __m256 sum_v[3];
        sum_v[0] = _mm256_setzero_ps();
        sum_v[1] = _mm256_setzero_ps();
        sum_v[2] = _mm256_setzero_ps();

        sum_v[3] = _mm256_setzero_ps();
        sum_v[4] = _mm256_setzero_ps();
        sum_v[5] = _mm256_setzero_ps();

        __m256 in_v;

        in_v = _mm256_loadu_ps(input_temp0);

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3), in_v, sum_v[3]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 8),
            _mm256_loadu_ps(input_temp0 + WIDTH - 3),
            0xf8); // 1111 1000

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 8), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 8), in_v, sum_v[3]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + WIDTH + 5),
            _mm256_loadu_ps(input_temp0 + 2*WIDTH - 6),
            0xc0); // 1100 0000

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 16), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 16), in_v, sum_v[3]);

        in_v = _mm256_loadu_ps(input_temp0 + 2*WIDTH + 2);

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 24), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 24), in_v, sum_v[3]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 2*WIDTH + 10),
            _mm256_loadu_ps(input_temp0 + 3*WIDTH - 1),
            0xfe); // 1111 1110

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 32), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 32), in_v, sum_v[3]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 3*WIDTH + 7),
            _mm256_loadu_ps(input_temp0 + 4*WIDTH - 4),
            0xf8); // 1111 0000

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 40), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 40), in_v, sum_v[3]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 4*WIDTH + 4),
            _mm256_loadu_ps(input_temp0 + 5*WIDTH - 7),
            0x80); // 1000 0000

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 48), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 48), in_v, sum_v[3]);

        in_v = _mm256_loadu_ps(input_temp0 + 5*WIDTH + 1);

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 56), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 56), in_v, sum_v[3]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 5*WIDTH + 9),
            _mm256_loadu_ps(input_temp0 + 6*WIDTH - 2),
            0xfc); // 1111 1100

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 64), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 64), in_v, sum_v[3]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 6*WIDTH + 6),
            _mm256_loadu_ps(input_temp0 + 7*WIDTH - 5),
            0xe0); // 1110 0000

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 72), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 72), in_v, sum_v[3]);

        in_v = _mm256_loadu_ps(input_temp0 + 7*WIDTH + 3);

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 80), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 80), in_v, sum_v[3]);

        in_v = _mm256_loadu_ps(input_temp0 + 8*WIDTH);

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 88), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 88), in_v, sum_v[3]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 8*WIDTH + 8),
            _mm256_loadu_ps(input_temp0 + 9*WIDTH - 3),
            0xf8); // 1111 1000

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 96), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 96), in_v, sum_v[3]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp0 + 9*WIDTH + 5),
            _mm256_loadu_ps(input_temp0 + 10*WIDTH - 6),
            0xc0); // 1100 0000

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 104), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 104), in_v, sum_v[3]);

        in_v = _mm256_loadu_ps(input_temp0 + 10*WIDTH + 2);

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 112), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 112), in_v, sum_v[3]);

        in_v = _mm256_maskload_ps(input_temp0 + 10*WIDTH + 10, mask1);

        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp0 + 120), in_v, sum_v[0]);
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp3 + 120), in_v, sum_v[3]);

        _mm_prefetch((const char *)(input_temp1 + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 2*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 3*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 4*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 5*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 6*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 7*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 8*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 9*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp1 + 10*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);

        in_v = _mm256_loadu_ps(input_temp1);

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4), in_v, sum_v[4]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 8),
            _mm256_loadu_ps(input_temp1 + WIDTH - 3),
            0xf8); // 1111 1000

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 8), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 8), in_v, sum_v[4]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + WIDTH + 5),
            _mm256_loadu_ps(input_temp1 + 2*WIDTH - 6),
            0xc0); // 1100 0000

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 16), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 16), in_v, sum_v[4]);

        in_v = _mm256_loadu_ps(input_temp1 + 2*WIDTH + 2);

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 24), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 24), in_v, sum_v[4]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 2*WIDTH + 10),
            _mm256_loadu_ps(input_temp1 + 3*WIDTH - 1),
            0xfe); // 1111 1110

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 32), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 32), in_v, sum_v[4]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 3*WIDTH + 7),
            _mm256_loadu_ps(input_temp1 + 4*WIDTH - 4),
            0xf8); // 1111 0000

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 40), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 40), in_v, sum_v[4]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 4*WIDTH + 4),
            _mm256_loadu_ps(input_temp1 + 5*WIDTH - 7),
            0x80); // 1000 0000

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 48), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 48), in_v, sum_v[4]);

        in_v = _mm256_loadu_ps(input_temp1 + 5*WIDTH + 1);

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 56), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 56), in_v, sum_v[4]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 5*WIDTH + 9),
            _mm256_loadu_ps(input_temp1 + 6*WIDTH - 2),
            0xfc); // 1111 1100

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 64), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 64), in_v, sum_v[4]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 6*WIDTH + 6),
            _mm256_loadu_ps(input_temp1 + 7*WIDTH - 5),
            0xe0); // 1110 0000

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 72), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 72), in_v, sum_v[4]);

        in_v = _mm256_loadu_ps(input_temp1 + 7*WIDTH + 3);

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 80), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 80), in_v, sum_v[4]);

        in_v = _mm256_loadu_ps(input_temp1 + 8*WIDTH);

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 88), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 88), in_v, sum_v[4]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 8*WIDTH + 8),
            _mm256_loadu_ps(input_temp1 + 9*WIDTH - 3),
            0xf8); // 1111 1000

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 96), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 96), in_v, sum_v[4]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp1 + 9*WIDTH + 5),
            _mm256_loadu_ps(input_temp1 + 10*WIDTH - 6),
            0xc0); // 1100 0000

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 104), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 104), in_v, sum_v[4]);

        in_v = _mm256_loadu_ps(input_temp1 + 10*WIDTH + 2);

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 112), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 112), in_v, sum_v[4]);

        in_v = _mm256_maskload_ps(input_temp1 + 10*WIDTH + 10, mask1);

        sum_v[1] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp1 + 120), in_v, sum_v[1]);
        sum_v[4] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp4 + 120), in_v, sum_v[4]);

        _mm_prefetch((const char *)(input_temp2 + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 2*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 3*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 4*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 5*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 6*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 7*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 8*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 9*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);
        _mm_prefetch((const char *)(input_temp2 + 10*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0);

        in_v = _mm256_loadu_ps(input_temp2);

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5), in_v, sum_v[5]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 8),
            _mm256_loadu_ps(input_temp2 + WIDTH - 3),
            0xf8); // 1111 1000

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 8), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 8), in_v, sum_v[5]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + WIDTH + 5),
            _mm256_loadu_ps(input_temp2 + 2*WIDTH - 6),
            0xc0); // 1100 0000

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 16), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 16), in_v, sum_v[5]);

        in_v = _mm256_loadu_ps(input_temp2 + 2*WIDTH + 2);

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 24), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 24), in_v, sum_v[5]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 2*WIDTH + 10),
            _mm256_loadu_ps(input_temp2 + 3*WIDTH - 1),
            0xfe); // 1111 1110

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 32), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 32), in_v, sum_v[5]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 3*WIDTH + 7),
            _mm256_loadu_ps(input_temp2 + 4*WIDTH - 4),
            0xf8); // 1111 0000

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 40), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 40), in_v, sum_v[5]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 4*WIDTH + 4),
            _mm256_loadu_ps(input_temp2 + 5*WIDTH - 7),
            0x80); // 1000 0000

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 48), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 48), in_v, sum_v[5]);

        in_v = _mm256_loadu_ps(input_temp2 + 5*WIDTH + 1);

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 56), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 56), in_v, sum_v[5]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 5*WIDTH + 9),
            _mm256_loadu_ps(input_temp2 + 6*WIDTH - 2),
            0xfc); // 1111 1100

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 64), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 64), in_v, sum_v[5]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 6*WIDTH + 6),
            _mm256_loadu_ps(input_temp2 + 7*WIDTH - 5),
            0xe0); // 1110 0000

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 72), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 72), in_v, sum_v[5]);

        in_v = _mm256_loadu_ps(input_temp2 + 7*WIDTH + 3);

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 80), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 80), in_v, sum_v[5]);

        in_v = _mm256_loadu_ps(input_temp2 + 8*WIDTH);

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 88), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 88), in_v, sum_v[5]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 8*WIDTH + 8),
            _mm256_loadu_ps(input_temp2 + 9*WIDTH - 3),
            0xf8); // 1111 1000

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 96), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 96), in_v, sum_v[5]);

        in_v = _mm256_blend_ps(
            _mm256_loadu_ps(input_temp2 + 9*WIDTH + 5),
            _mm256_loadu_ps(input_temp2 + 10*WIDTH - 6),
            0xc0); // 1100 0000

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 104), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 104), in_v, sum_v[5]);

        in_v = _mm256_loadu_ps(input_temp2 + 10*WIDTH + 2);

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 112), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 112), in_v, sum_v[5]);

        in_v = _mm256_maskload_ps(input_temp2 + 10*WIDTH + 10, mask1);

        sum_v[2] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp2 + 120), in_v, sum_v[2]);
        sum_v[5] = _mm256_fmadd_ps(_mm256_load_ps(weight_temp5 + 120), in_v, sum_v[5]);

        _mm256_store_ps(sum_temp, _mm256_add_ps(_mm256_add_ps(sum_v[0], sum_v[1]), sum_v[2]));
        float sum = 0;
        for (int i = 0; i < 8; ++i) {
          sum += sum_temp[i];
        }

        output[(out_channel * WOUT + output_row) * WOUT + output_col] = sum;

        _mm256_store_ps(sum_temp, _mm256_add_ps(_mm256_add_ps(sum_v[3], sum_v[4]), sum_v[5]));
        sum = 0;
        for (int i = 0; i < 8; ++i) {
          sum += sum_temp[i];
        }

        output[((out_channel + 1) * WOUT + output_row) * WOUT + output_col] = sum;
      } // for each output col
    } // for each output row
  } // for each out channel
}

// output channel unrolled twice
// output column unrolled twice
// input channel fully unrolled
// use permutation instructions to reduce # of load instructions
inline void conv1_ver4(const float *weight, const float *input, float *output)
{
  const int M = 96;
  const int WIDTH = 227;
  const int STRIDE = 4;
  const int K = 11;
  const int WOUT = (WIDTH - K)/STRIDE + 1; // 55

  // JSP: AlexNet conv1
  // Input: 3 x 227 x 227 => 201 KB per channel, 604 KB total
  // Output: 96 x 55 x 55 => 12 KB per channel, 1.1 MB total
  // Weight: 96 x 3 x 11 x 11 => 0.5 KB per channel pair, 1.5 KB per output channel, 45 KB per input channel, 136 KB total
  //         No matter what we do, there's no reuse on weight across different channels (only reuse is within a channel pair)
  // FLOPS: 2 x 96 x 3 x 55 x 55 x 11 x 11 = 211 MFLOPS
  //
  // Approach 1: stream output channel
  //   For each output channel, read inputs from LLC (604 KB total) and read weights from memory
  //   Total memory access: input 604 KB, output 1.1 MB, weight 136 KB
  //   Total LLC access: input 96 x 604 KB
  //
  // Approach 2: stream input channel
  //   For each input channel, read outputs from LLC (1.1 MB total) and read weights from memory
  //   Total memory access: input 604 KB, output 1.1 MB, weight 136 KB
  //   Total LLC access: output 2 x 1.1 MB
  //
  // Approach 3: blocking
  //   For each input channel, read 1/8 of outputs from L2 (1.1/8 MB total) and read weights from memory
  //   Total memory access: input 604 KB, output 1.1 MB, weight 136 KB
  //   Total LLC access: input 7 x 604 KB

  const int WOUT_BLOCK = 55;
  const int KERNEL_SIZE_ALIGNED = 128; // K*K

  int mask1_temp[8] = { -1, 0, 0, 0, 0, 0, 0, 0 };
  __m256i mask1 = _mm256_load_si256((__m256i *)mask1_temp);

  for (int out_channel = 0; out_channel < M; out_channel += 2) {
    __declspec(aligned(64)) float sum_temp0[8], sum_temp1[8], sum_temp2[8], sum_temp3[8];
    for (int output_row = 0; output_row < WOUT; ++output_row) {
      for (int output_col = 0; output_col < WOUT - 1; output_col += 2) {
        // in-channel loop fully-unrolled
        const float *input_temp0 = input + (output_row * WIDTH + output_col) * STRIDE;
        const float *input_temp1 = input_temp0 + WIDTH * WIDTH;
        const float *input_temp2 = input_temp0 + 2 * WIDTH * WIDTH;

        // in-channel loop fully-unrolled, out-channel loop unrolled twice
        const float *weight_temp0 = weight + 3 * out_channel * KERNEL_SIZE_ALIGNED;
        const float *weight_temp1 = weight_temp0 + KERNEL_SIZE_ALIGNED;
        const float *weight_temp2 = weight_temp0 + 2 * KERNEL_SIZE_ALIGNED;

        const float *weight_temp3 = weight + 3 * (out_channel + 1) * KERNEL_SIZE_ALIGNED;
        const float *weight_temp4 = weight_temp3 + KERNEL_SIZE_ALIGNED;
        const float *weight_temp5 = weight_temp3 + 2 * KERNEL_SIZE_ALIGNED;

        const int PREFETCH_DISTANCE = 16;

        __m256 sum_v[12];
        sum_v[0] = _mm256_setzero_ps(); // 1st in, 1st out, 1st col
        sum_v[1] = _mm256_setzero_ps();
        sum_v[2] = _mm256_setzero_ps();

        sum_v[3] = _mm256_setzero_ps(); // 1st in, 2nd out, 1st col
        sum_v[4] = _mm256_setzero_ps();
        sum_v[5] = _mm256_setzero_ps();

        sum_v[6] = _mm256_setzero_ps(); // 1st in, 1st out, 2nd col
        sum_v[7] = _mm256_setzero_ps();
        sum_v[8] = _mm256_setzero_ps();

        sum_v[9] = _mm256_setzero_ps(); // 1st in, 2nd out, 2nd col
        sum_v[10] = _mm256_setzero_ps();
        sum_v[11] = _mm256_setzero_ps();

        __m256 in0_v, in1_v;
        __m256 w0_v, w1_v;
        __m256 temp0_v, temp1_v, temp2_v, temp3_v;

#define MY_FMADD(w0, w1, offset, s0, s1, s2, s3) \
        w0_v = _mm256_load_ps(w0 + offset); /* weight for first out */ \
        w1_v = _mm256_load_ps(w1 + offset); /* weight for second out */ \
        s0 = _mm256_fmadd_ps(w0_v, in0_v, s0); /* 1st out, 1st col */\
        s1 = _mm256_fmadd_ps(w1_v, in0_v, s1); /* 2nd out, 1st col */\
        s2 = _mm256_fmadd_ps(w0_v, in1_v, s2); /* 1st out, 2nd col */\
        s3 = _mm256_fmadd_ps(w1_v, in1_v, s3); /* 2nd out, 2nd col */

#define MY_CONVOLVE(input_temp, w0, w1, s0, s1, s2, s3) \
        _mm_prefetch((const char *)(input_temp + PREFETCH_DISTANCE), _MM_HINT_T0); \
        _mm_prefetch((const char *)(input_temp + WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0); \
        _mm_prefetch((const char *)(input_temp + 2*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0); \
        _mm_prefetch((const char *)(input_temp + 3*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0); \
        _mm_prefetch((const char *)(input_temp + 4*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0); \
        _mm_prefetch((const char *)(input_temp + 5*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0); \
        _mm_prefetch((const char *)(input_temp + 6*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0); \
        _mm_prefetch((const char *)(input_temp + 7*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0); \
        _mm_prefetch((const char *)(input_temp + 8*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0); \
        _mm_prefetch((const char *)(input_temp + 9*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0); \
        _mm_prefetch((const char *)(input_temp + 10*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0); \
\
        in0_v = _mm256_loadu_ps(input_temp); \
        temp0_v = _mm256_loadu_ps(input_temp + 8); \
        in1_v = _mm256_permute2f128_ps(in0_v, temp0_v, 0x21); \
\
        MY_FMADD(w0, w1, 0, s0, s1, s2, s3); \
\
        temp1_v = _mm256_loadu_ps(input_temp + WIDTH - 3); \
        in0_v = _mm256_blend_ps( \
            temp0_v, \
            temp1_v, \
            0xf8); /* 1111 1000 */ \
        temp2_v = _mm256_loadu_ps(input_temp + WIDTH + 5); \
        in1_v = _mm256_blend_ps( \
            _mm256_permute2f128_ps(temp0_v, temp0_v, 0x11), \
            _mm256_permute2f128_ps(temp1_v, temp2_v, 0x21), \
            0xf8); /* 1111 1000 */ \
\
        MY_FMADD(w0, w1, 8, s0, s1, s2, s3); \
\
        temp1_v = _mm256_loadu_ps(input_temp + 2*WIDTH - 6); \
        in0_v = _mm256_blend_ps( \
            temp2_v, \
            temp1_v, \
            0xc0); /* 1100 0000 */ \
        temp0_v = _mm256_loadu_ps(input_temp + 2*WIDTH + 2); \
        in1_v = _mm256_blend_ps( \
            _mm256_loadu_ps(input_temp + WIDTH + 9), \
            _mm256_permute2f128_ps(temp0_v, temp0_v, 0x00), \
            0xc0); /* 1100 0000 */ \
        \
        MY_FMADD(w0, w1, 16, s0, s1, s2, s3); \
        \
        in0_v = temp0_v; \
        temp1_v = _mm256_loadu_ps(input_temp + 2*WIDTH + 10); \
        in1_v = _mm256_permute2f128_ps(temp0_v, temp1_v, 0x21); \
        \
        MY_FMADD(w0, w1, 24, s0, s1, s2, s3); \
        \
        temp0_v = _mm256_loadu_ps(input_temp + 3*WIDTH - 1); \
        in0_v = _mm256_blend_ps( \
            temp1_v, \
            temp0_v, \
            0xfe); /* 1111 1110 */ \
        temp2_v = _mm256_loadu_ps(input_temp + 3*WIDTH + 7); \
        in1_v = _mm256_blend_ps( \
            _mm256_permute2f128_ps(temp1_v, temp1_v, 0x11), \
            _mm256_permute2f128_ps(temp0_v, temp2_v, 0x21), \
            0xfe); /* 1111 1110 */ \
        \
        MY_FMADD(w0, w1, 32, s0, s1, s2, s3); \
        \
        temp1_v = _mm256_loadu_ps(input_temp + 4*WIDTH); \
        in0_v = _mm256_blend_ps( \
            temp2_v, \
            _mm256_permute2f128_ps(temp1_v, temp1_v, 0x00), \
            0xf8); /* 1111 0000 */ \
        in1_v = _mm256_blend_ps( \
            _mm256_permute2f128_ps(temp2_v, temp2_v, 0x11), \
            temp1_v, \
            0xf8); /* 1111 0000 */ \
        \
        MY_FMADD(w0, w1, 40, s0, s1, s2, s3); \
        \
        temp2_v = _mm256_loadu_ps(input_temp + 4*WIDTH + 8); \
        temp0_v = _mm256_loadu_ps(input_temp + 5*WIDTH - 7); \
        in0_v = _mm256_blend_ps( \
            _mm256_permute2f128_ps(temp1_v, temp2_v, 0x21), \
            temp0_v, \
            0x80); /* 1000 0000 */ \
        temp1_v = _mm256_loadu_ps(input_temp + 5*WIDTH + 1); \
        in1_v = _mm256_blend_ps( \
            temp2_v, \
            _mm256_permute2f128_ps(temp1_v, temp1_v, 0x00), \
            0x80); /* 1000 0000 */ \
        \
        MY_FMADD(w0, w1, 48, s0, s1, s2, s3); \
        \
        in0_v = temp1_v; \
        temp0_v = _mm256_loadu_ps(input_temp + 5*WIDTH + 9); \
        in1_v = _mm256_permute2f128_ps(temp1_v, temp0_v, 0x21); \
        \
        MY_FMADD(w0, w1, 56, s0, s1, s2, s3); \
        \
        temp1_v = _mm256_loadu_ps(input_temp + 6*WIDTH - 2); \
        in0_v = _mm256_blend_ps( \
            temp0_v, \
            temp1_v, \
            0xfc); /* 1111 1100 */ \
        temp2_v = _mm256_loadu_ps(input_temp + 6*WIDTH + 6); \
        in1_v = _mm256_blend_ps( \
            _mm256_permute2f128_ps(temp0_v, temp0_v, 0x11), \
            _mm256_permute2f128_ps(temp1_v, temp2_v, 0x21), \
            0xfc); /* 1111 1100 */ \
        \
        MY_FMADD(w0, w1, 64, s0, s1, s2, s3); \
        \
        temp1_v = _mm256_loadu_ps(input_temp + 7*WIDTH - 5); \
        in0_v = _mm256_blend_ps( \
            temp2_v, \
            temp1_v, \
            0xe0); /* 1110 0000 */ \
        temp0_v = _mm256_loadu_ps(input_temp + 7*WIDTH + 3); \
        in1_v = _mm256_blend_ps( \
            _mm256_loadu_ps(input_temp + 6*WIDTH + 10), \
            _mm256_permute2f128_ps(temp0_v, temp0_v, 0x00), \
            0xe0); /* 1110 0000 */ \
        \
        MY_FMADD(w0, w1, 72, s0, s1, s2, s3); \
        \
        in0_v = temp0_v; \
        in1_v = _mm256_loadu_ps(input_temp + 7*WIDTH + 7); \
        \
        MY_FMADD(w0, w1, 80, s0, s1, s2, s3); \
        \
        in0_v = _mm256_loadu_ps(input_temp + 8*WIDTH); \
        temp0_v = _mm256_loadu_ps(input_temp + 8*WIDTH + 8); \
        in1_v = _mm256_permute2f128_ps(in0_v, temp0_v, 0x21); \
        \
        MY_FMADD(w0, w1, 88, s0, s1, s2, s3); \
        \
        temp1_v = _mm256_loadu_ps(input_temp + 9*WIDTH - 3); \
        in0_v = _mm256_blend_ps( \
            temp0_v, \
            temp1_v, \
            0xf8); /* 1111 1000 */ \
        temp2_v = _mm256_loadu_ps(input_temp + 9*WIDTH + 5); \
        in1_v = _mm256_blend_ps( \
            _mm256_permute2f128_ps(temp0_v, temp0_v, 0x11), \
            _mm256_permute2f128_ps(temp1_v, temp2_v, 0x21), \
            0xf8); /* 1111 1000 */ \
        \
        MY_FMADD(w0, w1, 96, s0, s1, s2, s3); \
        \
        temp1_v = _mm256_loadu_ps(input_temp + 10*WIDTH - 2); \
        in0_v = _mm256_blend_ps( \
            temp2_v, \
            _mm256_permute2f128_ps(temp1_v, temp1_v, 0x00), \
            0xc0); /* 1100 0000 */ \
        in1_v = _mm256_blend_ps( \
            _mm256_loadu_ps(input_temp + 9*WIDTH + 9), \
            temp1_v, \
            0xc0); /* 1100 0000 */ \
        \
        MY_FMADD(w0, w1, 104, s0, s1, s2, s3); \
        \
        temp0_v = _mm256_loadu_ps(input_temp + 10*WIDTH + 6); \
        in0_v = _mm256_permute2f128_ps(temp1_v, temp0_v, 0x21); \
        in1_v = temp0_v; \
        \
        MY_FMADD(w0, w1, 112, s0, s1, s2, s3); \
        \
        in0_v = _mm256_maskload_ps(input_temp + 10*WIDTH + 10, mask1); \
        in1_v = _mm256_maskload_ps(input_temp + 10*WIDTH + 14, mask1); \
        \
        MY_FMADD(w0, w1, 120, s0, s1, s2, s3);

        MY_CONVOLVE(input_temp0, weight_temp0, weight_temp3, sum_v[0], sum_v[3], sum_v[6], sum_v[9]);
        MY_CONVOLVE(input_temp1, weight_temp1, weight_temp4, sum_v[1], sum_v[4], sum_v[7], sum_v[10]);
        MY_CONVOLVE(input_temp2, weight_temp2, weight_temp5, sum_v[2], sum_v[5], sum_v[8], sum_v[11]);

        _mm256_store_ps(sum_temp0, _mm256_add_ps(_mm256_add_ps(sum_v[0], sum_v[1]), sum_v[2]));
        float sum0 = 0;
        for (int i = 0; i < 8; ++i) {
          sum0 += sum_temp0[i];
        }

        output[(out_channel * WOUT + output_row) * WOUT + output_col] = sum0;

        _mm256_store_ps(sum_temp1, _mm256_add_ps(_mm256_add_ps(sum_v[3], sum_v[4]), sum_v[5]));
        float sum1 = 0;
        for (int i = 0; i < 8; ++i) {
          sum1 += sum_temp1[i];
        }

        output[((out_channel + 1) * WOUT + output_row) * WOUT + output_col] = sum1;

        _mm256_store_ps(sum_temp2, _mm256_add_ps(_mm256_add_ps(sum_v[6], sum_v[7]), sum_v[8]));
        float sum2 = 0;
        for (int i = 0; i < 8; ++i) {
          sum2 += sum_temp2[i];
        }

        output[(out_channel * WOUT + output_row) * WOUT + output_col + 1] = sum2;

        _mm256_store_ps(sum_temp3, _mm256_add_ps(_mm256_add_ps(sum_v[9], sum_v[10]), sum_v[11]));
        float sum3 = 0;
        for (int i = 0; i < 8; ++i) {
          sum3 += sum_temp3[i];
        }

        output[((out_channel + 1) * WOUT + output_row) * WOUT + output_col + 1] = sum3;
      } // for each output col

      for (int output_col = WOUT - 1; output_col < WOUT; ++output_col) {
        const float *input_temp0 = input + (output_row * WIDTH + output_col) * STRIDE;
        const float *input_temp1 = input_temp0 + WIDTH * WIDTH;
        const float *input_temp2 = input_temp0 + 2 * WIDTH * WIDTH;

        const float *weight_temp0 = weight + 3 * out_channel * KERNEL_SIZE_ALIGNED;
        const float *weight_temp1 = weight_temp0 + KERNEL_SIZE_ALIGNED;
        const float *weight_temp2 = weight_temp0 + 2 * KERNEL_SIZE_ALIGNED;

        const float *weight_temp3 = weight + 3 * (out_channel + 1) * KERNEL_SIZE_ALIGNED;
        const float *weight_temp4 = weight_temp3 + KERNEL_SIZE_ALIGNED;
        const float *weight_temp5 = weight_temp3 + 2 * KERNEL_SIZE_ALIGNED;

        __m256 sum_v[6];
        sum_v[0] = _mm256_setzero_ps();
        sum_v[1] = _mm256_setzero_ps();
        sum_v[2] = _mm256_setzero_ps();

        sum_v[3] = _mm256_setzero_ps();
        sum_v[4] = _mm256_setzero_ps();
        sum_v[5] = _mm256_setzero_ps();

        __m256 in_v;

#undef MY_CONVOLVE
#define MY_CONVOLVE(input_temp, w0, w1) \
        in_v = _mm256_loadu_ps(input_temp); \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0), in_v, sum_v[0]); \
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(w1), in_v, sum_v[3]); \
        \
        in_v = _mm256_blend_ps( \
            _mm256_loadu_ps(input_temp + 8), \
            _mm256_loadu_ps(input_temp + WIDTH - 3), \
            0xf8); /* 1111 1000 */ \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 8), in_v, sum_v[0]); \
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(w1 + 8), in_v, sum_v[3]); \
        \
        in_v = _mm256_blend_ps( \
            _mm256_loadu_ps(input_temp + WIDTH + 5), \
            _mm256_loadu_ps(input_temp + 2*WIDTH - 6), \
            0xc0); /* 1100 0000 */ \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 16), in_v, sum_v[0]); \
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(w1 + 16), in_v, sum_v[3]); \
        \
        in_v = _mm256_loadu_ps(input_temp + 2*WIDTH + 2); \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 24), in_v, sum_v[0]); \
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(w1 + 24), in_v, sum_v[3]); \
        \
        in_v = _mm256_blend_ps( \
            _mm256_loadu_ps(input_temp + 2*WIDTH + 10), \
            _mm256_loadu_ps(input_temp + 3*WIDTH - 1), \
            0xfe); /* 1111 1110 */ \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 32), in_v, sum_v[0]); \
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(w1 + 32), in_v, sum_v[3]); \
        \
        in_v = _mm256_blend_ps( \
            _mm256_loadu_ps(input_temp + 3*WIDTH + 7), \
            _mm256_loadu_ps(input_temp + 4*WIDTH - 4), \
            0xf8); /* 1111 0000 */ \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 40), in_v, sum_v[0]); \
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(w1 + 40), in_v, sum_v[3]); \
        \
        in_v = _mm256_blend_ps( \
            _mm256_loadu_ps(input_temp + 4*WIDTH + 4), \
            _mm256_loadu_ps(input_temp + 5*WIDTH - 7), \
            0x80); /* 1000 0000 */ \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 48), in_v, sum_v[0]); \
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(w1 + 48), in_v, sum_v[3]); \
        \
        in_v = _mm256_loadu_ps(input_temp + 5*WIDTH + 1); \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 56), in_v, sum_v[0]); \
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(w1 + 56), in_v, sum_v[3]); \
        \
        in_v = _mm256_blend_ps( \
            _mm256_loadu_ps(input_temp + 5*WIDTH + 9), \
            _mm256_loadu_ps(input_temp + 6*WIDTH - 2), \
            0xfc); /* 1111 1100 */ \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 64), in_v, sum_v[0]); \
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(w1 + 64), in_v, sum_v[3]); \
        \
        in_v = _mm256_blend_ps( \
            _mm256_loadu_ps(input_temp + 6*WIDTH + 6), \
            _mm256_loadu_ps(input_temp + 7*WIDTH - 5), \
            0xe0); /* 1110 0000 */ \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 72), in_v, sum_v[0]); \
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(w1 + 72), in_v, sum_v[3]); \
        \
        in_v = _mm256_loadu_ps(input_temp + 7*WIDTH + 3); \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 80), in_v, sum_v[0]); \
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(w1 + 80), in_v, sum_v[3]); \
        \
        in_v = _mm256_loadu_ps(input_temp + 8*WIDTH); \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 88), in_v, sum_v[0]); \
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(w1 + 88), in_v, sum_v[3]); \
        \
        in_v = _mm256_blend_ps( \
            _mm256_loadu_ps(input_temp + 8*WIDTH + 8), \
            _mm256_loadu_ps(input_temp + 9*WIDTH - 3), \
            0xf8); /* 1111 1000 */ \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 96), in_v, sum_v[0]); \
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(w1 + 96), in_v, sum_v[3]); \
        \
        in_v = _mm256_blend_ps( \
            _mm256_loadu_ps(input_temp + 9*WIDTH + 5), \
            _mm256_loadu_ps(input_temp + 10*WIDTH - 6), \
            0xc0); /* 1100 0000 */ \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 104), in_v, sum_v[0]); \
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(w1 + 104), in_v, sum_v[3]); \
        \
        in_v = _mm256_loadu_ps(input_temp + 10*WIDTH + 2); \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 112), in_v, sum_v[0]); \
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(w1 + 112), in_v, sum_v[3]); \
        \
        in_v = _mm256_maskload_ps(input_temp + 10*WIDTH + 10, mask1); \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 120), in_v, sum_v[0]); \
        sum_v[3] = _mm256_fmadd_ps(_mm256_load_ps(w1 + 120), in_v, sum_v[3]);

        MY_CONVOLVE(input_temp0, weight_temp0, weight_temp3);
        MY_CONVOLVE(input_temp1, weight_temp1, weight_temp4);
        MY_CONVOLVE(input_temp2, weight_temp2, weight_temp5);

        _mm256_store_ps(sum_temp0, _mm256_add_ps(_mm256_add_ps(sum_v[0], sum_v[1]), sum_v[2]));
        float sum0 = 0;
        for (int i = 0; i < 8; ++i) {
          sum0 += sum_temp0[i];
        }

        output[(out_channel * WOUT + output_row) * WOUT + output_col] = sum0;

        _mm256_store_ps(sum_temp1, _mm256_add_ps(_mm256_add_ps(sum_v[3], sum_v[4]), sum_v[5]));
        float sum1 = 0;
        for (int i = 0; i < 8; ++i) {
          sum1 += sum_temp1[i];
        }

        output[((out_channel + 1) * WOUT + output_row) * WOUT + output_col] = sum1;
      } // for each output col
    } // for each output row
  } // for each out channel

#undef MY_FMADD
#undef MY_CONVOLVE
}

// output column unrolled 4 times
// input channel fully unrolled
// use permutation instructions to reduce # of load instructions
inline void conv1_ver5(const float *weight, const float *input, float *output)
{
  const int M = 96;
  const int WIDTH = 227;
  const int STRIDE = 4;
  const int K = 11;
  const int WOUT = (WIDTH - K)/STRIDE + 1; // 55

  // JSP: AlexNet conv1
  // Input: 3 x 227 x 227 => 201 KB per channel, 604 KB total
  // Output: 96 x 55 x 55 => 12 KB per channel, 1.1 MB total
  // Weight: 96 x 3 x 11 x 11 => 0.5 KB per channel pair, 1.5 KB per output channel, 45 KB per input channel, 136 KB total
  //         No matter what we do, there's no reuse on weight across different channels (only reuse is within a channel pair)
  // FLOPS: 2 x 96 x 3 x 55 x 55 x 11 x 11 = 211 MFLOPS
  //
  // Approach 1: stream output channel
  //   For each output channel, read inputs from LLC (604 KB total) and read weights from memory
  //   Total memory access: input 604 KB, output 1.1 MB, weight 136 KB
  //   Total LLC access: input 96 x 604 KB
  //
  // Approach 2: stream input channel
  //   For each input channel, read outputs from LLC (1.1 MB total) and read weights from memory
  //   Total memory access: input 604 KB, output 1.1 MB, weight 136 KB
  //   Total LLC access: output 2 x 1.1 MB
  //
  // Approach 3: blocking
  //   For each input channel, read 1/8 of outputs from L2 (1.1/8 MB total) and read weights from memory
  //   Total memory access: input 604 KB, output 1.1 MB, weight 136 KB
  //   Total LLC access: input 7 x 604 KB

  const int WOUT_BLOCK = 55;
  const int KERNEL_SIZE_ALIGNED = 128; // K*K

  __declspec(aligned(64)) float sum_temp0[8], sum_temp1[8], sum_temp2[8], sum_temp3[8];
  __declspec(aligned(64)) int mask1_temp[8] = { -1, 0, 0, 0, 0, 0, 0, 0 };
  __m256i mask1 = _mm256_load_si256((__m256i *)mask1_temp);

  for (int out_channel = 0; out_channel < M; ++out_channel) {
    for (int output_row = 0; output_row < WOUT; ++output_row) {
      for (int output_col = 0; output_col < WOUT - 3; output_col += 4) {
        __m256 sum_v[4];
        sum_v[0] = _mm256_setzero_ps(); // 1st col
        sum_v[1] = _mm256_setzero_ps(); // 2nd col
        sum_v[2] = _mm256_setzero_ps(); // 3rd col
        sum_v[3] = _mm256_setzero_ps(); // 4th col

        for (int in_channel = 0; in_channel < 3; ++in_channel) {
          const float *input_temp = input + (output_row * WIDTH + output_col) * STRIDE + in_channel * WIDTH * WIDTH;
          const float *weight_temp = weight + (3 * out_channel + in_channel) * KERNEL_SIZE_ALIGNED;

          const int PREFETCH_DISTANCE = 16;

          __m256 in0_v, in1_v, in2_v, in3_v;
          __m256 w_v;
          __m256
            temp0_v, temp1_v, temp2_v, temp3_v,
            temp4_v, temp5_v, temp6_v, temp7_v,
            temp8_v, temp9_v;

#define MY_FMADD(w, offset, s0, s1, s2, s3) \
          w_v = _mm256_load_ps(w + offset); \
          s0 = _mm256_fmadd_ps(w_v, in0_v, s0); /* 1st col */ \
          s1 = _mm256_fmadd_ps(w_v, in1_v, s1); /* 2nd col */ \
          s2 = _mm256_fmadd_ps(w_v, in2_v, s2); /* 3rd col */ \
          s3 = _mm256_fmadd_ps(w_v, in3_v, s3); /* 4th col */

#define MY_CONVOLVE(input_temp, w, s0, s1, s2, s3) \
          _mm_prefetch((const char *)(input_temp + PREFETCH_DISTANCE), _MM_HINT_T0); \
          _mm_prefetch((const char *)(input_temp + WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0); \
          _mm_prefetch((const char *)(input_temp + 2*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0); \
          _mm_prefetch((const char *)(input_temp + 3*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0); \
          _mm_prefetch((const char *)(input_temp + 4*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0); \
          _mm_prefetch((const char *)(input_temp + 5*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0); \
          _mm_prefetch((const char *)(input_temp + 6*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0); \
          _mm_prefetch((const char *)(input_temp + 7*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0); \
          _mm_prefetch((const char *)(input_temp + 8*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0); \
          _mm_prefetch((const char *)(input_temp + 9*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0); \
          _mm_prefetch((const char *)(input_temp + 10*WIDTH + PREFETCH_DISTANCE), _MM_HINT_T0); \
          \
          temp0_v = _mm256_loadu_ps(input_temp); \
          temp1_v = _mm256_loadu_ps(input_temp + 8); \
          temp2_v = _mm256_loadu_ps(input_temp + 16); \
          temp3_v = _mm256_permute2f128_ps(temp1_v, temp2_v, 0x21); \
          \
          in0_v = temp0_v; \
          in1_v = _mm256_permute2f128_ps(temp0_v, temp1_v, 0x21); \
          in2_v = temp1_v; \
          in3_v = temp3_v; \
          \
          MY_FMADD(w, 0, s0, s1, s2, s3); \
          \
          temp4_v = _mm256_loadu_ps(input_temp + WIDTH - 3); \
          temp5_v = _mm256_loadu_ps(input_temp + WIDTH + 5); \
          temp6_v = _mm256_loadu_ps(input_temp + WIDTH + 13); \
          temp7_v = _mm256_permute2f128_ps(temp5_v, temp6_v, 0x21); \
          \
          in0_v = _mm256_blend_ps( \
              temp1_v, \
              temp4_v, \
              0xf8); /* 1111 1000 */ \
          in1_v = _mm256_blend_ps( \
              temp3_v, \
              _mm256_permute2f128_ps(temp4_v, temp5_v, 0x21), \
              0xf8); /* 1111 1000 */ \
          in2_v = _mm256_blend_ps( \
              temp2_v, \
              temp5_v, \
              0xf8); \
          in3_v = _mm256_blend_ps( \
              _mm256_permute2f128_ps(temp2_v, temp2_v, 0x11), \
              temp7_v, \
              0xf8); \
          \
          MY_FMADD(w, 8, s0, s1, s2, s3); \
          \
          temp0_v = _mm256_loadu_ps(input_temp + 2*WIDTH - 2); \
          temp1_v = _mm256_loadu_ps(input_temp + 2*WIDTH + 6); \
          temp2_v = _mm256_loadu_ps(input_temp + 2*WIDTH + 14); \
          temp8_v = _mm256_permute2f128_ps(temp0_v, temp1_v, 0x21); \
          temp9_v = _mm256_permute2f128_ps(temp1_v, temp2_v, 0x21); \
          \
          in0_v = _mm256_blend_ps( \
              temp5_v, \
              _mm256_permute2f128_ps(temp0_v, temp0_v, 0x00), \
              0xc0); /* 1100 0000 */ \
          in1_v = _mm256_blend_ps( \
              temp7_v, \
              temp0_v, \
              0xc0); /* 1100 0000 */ \
          in2_v = _mm256_blend_ps( \
              temp6_v, \
              temp8_v, \
              0xc0); \
          in3_v = _mm256_blend_ps( \
              _mm256_loadu_ps(input_temp + WIDTH + 17), \
              temp1_v, \
              0xc0); \
          \
          MY_FMADD(w, 16, s0, s1, s2, s3); \
          \
          in0_v = temp8_v; \
          in1_v = temp1_v; \
          in2_v = temp9_v; \
          in3_v = temp2_v; \
          \
          MY_FMADD(w, 24, s0, s1, s2, s3); \
          \
          temp4_v = _mm256_loadu_ps(input_temp + 3*WIDTH - 1); \
          temp5_v = _mm256_loadu_ps(input_temp + 3*WIDTH + 7); \
          temp6_v = _mm256_loadu_ps(input_temp + 3*WIDTH + 15); \
          temp7_v = _mm256_permute2f128_ps(temp5_v, temp6_v, 0x21); \
          \
          in0_v = _mm256_blend_ps( \
              temp9_v, \
              temp4_v, \
              0xfe); /* 1111 1110 */ \
          in1_v = _mm256_blend_ps( \
              temp2_v, \
              _mm256_permute2f128_ps(temp4_v, temp5_v, 0x21), \
              0xfe); /* 1111 1110 */ \
          in2_v = _mm256_blend_ps( \
              _mm256_permute2f128_ps(temp2_v, temp2_v, 0x11), \
              temp5_v, \
              0xfe); \
          in3_v = _mm256_blend_ps( \
              _mm256_loadu_ps(input_temp + 2*WIDTH + 22), \
              temp7_v, \
              0xfe); \
          \
          MY_FMADD(w, 32, s0, s1, s2, s3); \
          \
          temp0_v = _mm256_loadu_ps(input_temp + 4*WIDTH); \
          temp1_v = _mm256_loadu_ps(input_temp + 4*WIDTH + 8); \
          temp2_v = _mm256_loadu_ps(input_temp + 4*WIDTH + 16); \
          temp3_v = _mm256_permute2f128_ps(temp0_v, temp1_v, 0x21); \
          \
          in0_v = _mm256_blend_ps( \
              temp5_v, \
              _mm256_permute2f128_ps(temp0_v, temp0_v, 0x00), \
              0xf8); /* 1111 0000 */ \
          in1_v = _mm256_blend_ps( \
              temp7_v, \
              temp0_v, \
              0xf8); /* 1111 0000 */ \
          in2_v = _mm256_blend_ps( \
              temp6_v, \
              temp3_v, \
              0xf8); \
          in3_v = _mm256_blend_ps( \
              _mm256_permute2f128_ps(temp6_v, temp6_v, 0x11), \
              temp1_v, \
              0xf8); \
          \
          MY_FMADD(w, 40, s0, s1, s2, s3); \
          \
          temp4_v = _mm256_loadu_ps(input_temp + 5*WIDTH - 3); \
          temp5_v = _mm256_loadu_ps(input_temp + 5*WIDTH + 5); \
          temp6_v = _mm256_loadu_ps(input_temp + 5*WIDTH + 13); \
          temp8_v = _mm256_permute2f128_ps(temp4_v, temp5_v, 0x21); \
          temp9_v = _mm256_permute2f128_ps(temp5_v, temp6_v, 0x21); \
          \
          in0_v = _mm256_blend_ps( \
              temp3_v, \
              _mm256_permute2f128_ps(temp4_v, temp4_v, 0x00), \
              0x80); /* 1000 0000 */ \
          in1_v = _mm256_blend_ps( \
              temp1_v, \
              temp4_v, \
              0x80); /* 1000 0000 */ \
          in2_v = _mm256_blend_ps( \
              _mm256_permute2f128_ps(temp1_v, temp2_v, 0x21), \
              temp8_v, \
              0x80); \
          in3_v = _mm256_blend_ps( \
              temp2_v, \
              temp5_v, \
              0x80); \
          \
          MY_FMADD(w, 48, s0, s1, s2, s3); \
          \
          in0_v = temp8_v; \
          in1_v = temp5_v; \
          in2_v = temp9_v; \
          in3_v = temp6_v; \
          \
          MY_FMADD(w, 56, s0, s1, s2, s3); \
          \
          temp0_v = _mm256_loadu_ps(input_temp + 6*WIDTH - 2); \
          temp1_v = _mm256_loadu_ps(input_temp + 6*WIDTH + 6); \
          temp2_v = _mm256_loadu_ps(input_temp + 6*WIDTH + 14); \
          temp3_v = _mm256_permute2f128_ps(temp1_v, temp2_v, 0x21); \
          \
          in0_v = _mm256_blend_ps( \
              temp9_v, \
              temp0_v, \
              0xfc); /* 1111 1100 */ \
          in1_v = _mm256_blend_ps( \
              temp6_v, \
              _mm256_permute2f128_ps(temp0_v, temp1_v, 0x21), \
              0xfc); /* 1111 1100 */ \
          in2_v = _mm256_blend_ps( \
              _mm256_permute2f128_ps(temp6_v, temp6_v, 0x11), \
              temp1_v, \
              0xfc); \
          in3_v = _mm256_blend_ps( \
              _mm256_loadu_ps(input_temp + 5*WIDTH + 21), \
              temp3_v, \
              0xfc); \
          \
          MY_FMADD(w, 64, s0, s1, s2, s3); \
          \
          temp4_v = _mm256_loadu_ps(input_temp + 7*WIDTH - 1); \
          temp5_v = _mm256_loadu_ps(input_temp + 7*WIDTH + 7); \
          temp6_v = _mm256_loadu_ps(input_temp + 7*WIDTH + 15); \
          temp7_v = _mm256_permute2f128_ps(temp4_v, temp5_v, 0x21); \
          \
          in0_v = _mm256_blend_ps( \
              temp1_v, \
              _mm256_permute2f128_ps(temp4_v, temp4_v, 0x00), \
              0xe0); /* 1110 0000 */ \
          in1_v = _mm256_blend_ps( \
              temp3_v, \
              temp4_v, \
              0xe0); /* 1110 0000 */ \
          in2_v = _mm256_blend_ps( \
              temp2_v, \
              temp7_v, \
              0xe0); \
          in3_v = _mm256_blend_ps( \
              _mm256_loadu_ps(input_temp + 6*WIDTH + 18), \
              temp5_v, \
              0xe0); \
          \
          MY_FMADD(w, 72, s0, s1, s2, s3); \
          \
          in0_v = temp7_v; \
          in1_v = temp5_v; \
          in2_v = _mm256_permute2f128_ps(temp5_v, temp6_v, 0x21); \
          in3_v = temp6_v; \
          \
          MY_FMADD(w, 80, s0, s1, s2, s3); \
          \
          temp0_v = _mm256_loadu_ps(input_temp + 8*WIDTH); \
          temp1_v = _mm256_loadu_ps(input_temp + 8*WIDTH + 8); \
          temp2_v = _mm256_loadu_ps(input_temp + 8*WIDTH + 16); \
          temp3_v = _mm256_permute2f128_ps(temp1_v, temp2_v, 0x21); \
          \
          in0_v = temp0_v; \
          in1_v = _mm256_permute2f128_ps(temp0_v, temp1_v, 0x21); \
          in2_v = temp1_v; \
          in3_v = temp3_v; \
          \
          MY_FMADD(w, 88, s0, s1, s2, s3); \
          \
          temp4_v = _mm256_loadu_ps(input_temp + 9*WIDTH - 3); \
          temp5_v = _mm256_loadu_ps(input_temp + 9*WIDTH + 5); \
          temp6_v = _mm256_loadu_ps(input_temp + 9*WIDTH + 13); \
          temp7_v = _mm256_permute2f128_ps(temp5_v, temp6_v, 0x21); \
          \
          in0_v = _mm256_blend_ps( \
              temp1_v, \
              temp4_v, \
              0xf8); /* 1111 1000 */ \
          in1_v = _mm256_blend_ps( \
              temp3_v, \
              _mm256_permute2f128_ps(temp4_v, temp5_v, 0x21), \
              0xf8); /* 1111 1000 */ \
          in2_v = _mm256_blend_ps( \
              temp2_v, \
              temp5_v, \
              0xf8); \
          in3_v = _mm256_blend_ps( \
              _mm256_permute2f128_ps(temp2_v, temp2_v, 0x11), \
              temp7_v, \
              0xf8); \
          \
          MY_FMADD(w, 96, s0, s1, s2, s3); \
          \
          temp0_v = _mm256_loadu_ps(input_temp + 10*WIDTH - 2); \
          temp1_v = _mm256_loadu_ps(input_temp + 10*WIDTH + 6); \
          temp2_v = _mm256_loadu_ps(input_temp + 10*WIDTH + 14); \
          temp8_v = _mm256_permute2f128_ps(temp0_v, temp1_v, 0x21); \
          temp9_v = _mm256_permute2f128_ps(temp1_v, temp2_v, 0x21); \
          \
          in0_v = _mm256_blend_ps( \
              temp5_v, \
              _mm256_permute2f128_ps(temp0_v, temp0_v, 0x00), \
              0xc0); /* 1100 0000 */ \
          in1_v = _mm256_blend_ps( \
              temp7_v, \
              temp0_v, \
              0xc0); /* 1100 0000 */ \
          in2_v = _mm256_blend_ps( \
              temp6_v, \
              temp8_v, \
              0xc0); \
          in3_v = _mm256_blend_ps( \
              _mm256_loadu_ps(input_temp + 9*WIDTH + 17), \
              temp1_v, \
              0xc0); \
          \
          MY_FMADD(w, 104, s0, s1, s2, s3); \
          \
          in0_v = temp8_v; \
          in1_v = temp1_v; \
          in2_v = temp9_v; \
          in3_v = temp2_v; \
          \
          MY_FMADD(w, 112, s0, s1, s2, s3); \
          \
          in0_v = _mm256_maskload_ps(input_temp + 10*WIDTH + 10, mask1); \
          in1_v = _mm256_maskload_ps(input_temp + 10*WIDTH + 14, mask1); \
          in2_v = _mm256_maskload_ps(input_temp + 10*WIDTH + 18, mask1); \
          in3_v = _mm256_maskload_ps(input_temp + 10*WIDTH + 22, mask1); \
          \
          MY_FMADD(w, 120, s0, s1, s2, s3);

          MY_CONVOLVE(input_temp, weight_temp, sum_v[0], sum_v[1], sum_v[2], sum_v[3]);
        }

        _mm256_store_ps(sum_temp0, sum_v[0]);
        float sum0 = 0;
        for (int i = 0; i < 8; ++i) {
          sum0 += sum_temp0[i];
        }

        output[(out_channel * WOUT + output_row) * WOUT + output_col] = sum0;

        _mm256_store_ps(sum_temp1, sum_v[1]);
        float sum1 = 0;
        for (int i = 0; i < 8; ++i) {
          sum1 += sum_temp1[i];
        }

        output[(out_channel * WOUT + output_row) * WOUT + output_col + 1] = sum1;

        _mm256_store_ps(sum_temp2, sum_v[2]);
        float sum2 = 0;
        for (int i = 0; i < 8; ++i) {
          sum2 += sum_temp2[i];
        }

        output[(out_channel * WOUT + output_row) * WOUT + output_col + 2] = sum2;

        _mm256_store_ps(sum_temp3, sum_v[3]);
        float sum3 = 0;
        for (int i = 0; i < 8; ++i) {
          sum3 += sum_temp3[i];
        }

        output[(out_channel * WOUT + output_row) * WOUT + output_col + 3] = sum3;
      } // for each output col

      for (int output_col = WOUT - 3; output_col < WOUT; ++output_col) {
        const float *input_temp0 = input + (output_row * WIDTH + output_col) * STRIDE;
        const float *input_temp1 = input_temp0 + WIDTH * WIDTH;
        const float *input_temp2 = input_temp0 + 2 * WIDTH * WIDTH;

        const float *weight_temp0 = weight + 3 * out_channel * KERNEL_SIZE_ALIGNED;
        const float *weight_temp1 = weight_temp0 + KERNEL_SIZE_ALIGNED;
        const float *weight_temp2 = weight_temp0 + 2 * KERNEL_SIZE_ALIGNED;

        __m256 sum_v[6];
        sum_v[0] = _mm256_setzero_ps();

        __m256 in_v;

#undef MY_CONVOLVE
#define MY_CONVOLVE(input_temp, w0) \
        in_v = _mm256_loadu_ps(input_temp); \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0), in_v, sum_v[0]); \
        \
        in_v = _mm256_blend_ps( \
            _mm256_loadu_ps(input_temp + 8), \
            _mm256_loadu_ps(input_temp + WIDTH - 3), \
            0xf8); /* 1111 1000 */ \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 8), in_v, sum_v[0]); \
        \
        in_v = _mm256_blend_ps( \
            _mm256_loadu_ps(input_temp + WIDTH + 5), \
            _mm256_loadu_ps(input_temp + 2*WIDTH - 6), \
            0xc0); /* 1100 0000 */ \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 16), in_v, sum_v[0]); \
        \
        in_v = _mm256_loadu_ps(input_temp + 2*WIDTH + 2); \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 24), in_v, sum_v[0]); \
        \
        in_v = _mm256_blend_ps( \
            _mm256_loadu_ps(input_temp + 2*WIDTH + 10), \
            _mm256_loadu_ps(input_temp + 3*WIDTH - 1), \
            0xfe); /* 1111 1110 */ \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 32), in_v, sum_v[0]); \
        \
        in_v = _mm256_blend_ps( \
            _mm256_loadu_ps(input_temp + 3*WIDTH + 7), \
            _mm256_loadu_ps(input_temp + 4*WIDTH - 4), \
            0xf8); /* 1111 0000 */ \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 40), in_v, sum_v[0]); \
        \
        in_v = _mm256_blend_ps( \
            _mm256_loadu_ps(input_temp + 4*WIDTH + 4), \
            _mm256_loadu_ps(input_temp + 5*WIDTH - 7), \
            0x80); /* 1000 0000 */ \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 48), in_v, sum_v[0]); \
        \
        in_v = _mm256_loadu_ps(input_temp + 5*WIDTH + 1); \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 56), in_v, sum_v[0]); \
        \
        in_v = _mm256_blend_ps( \
            _mm256_loadu_ps(input_temp + 5*WIDTH + 9), \
            _mm256_loadu_ps(input_temp + 6*WIDTH - 2), \
            0xfc); /* 1111 1100 */ \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 64), in_v, sum_v[0]); \
        \
        in_v = _mm256_blend_ps( \
            _mm256_loadu_ps(input_temp + 6*WIDTH + 6), \
            _mm256_loadu_ps(input_temp + 7*WIDTH - 5), \
            0xe0); /* 1110 0000 */ \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 72), in_v, sum_v[0]); \
        \
        in_v = _mm256_loadu_ps(input_temp + 7*WIDTH + 3); \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 80), in_v, sum_v[0]); \
        \
        in_v = _mm256_loadu_ps(input_temp + 8*WIDTH); \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 88), in_v, sum_v[0]); \
        \
        in_v = _mm256_blend_ps( \
            _mm256_loadu_ps(input_temp + 8*WIDTH + 8), \
            _mm256_loadu_ps(input_temp + 9*WIDTH - 3), \
            0xf8); /* 1111 1000 */ \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 96), in_v, sum_v[0]); \
        \
        in_v = _mm256_blend_ps( \
            _mm256_loadu_ps(input_temp + 9*WIDTH + 5), \
            _mm256_loadu_ps(input_temp + 10*WIDTH - 6), \
            0xc0); /* 1100 0000 */ \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 104), in_v, sum_v[0]); \
        \
        in_v = _mm256_loadu_ps(input_temp + 10*WIDTH + 2); \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 112), in_v, sum_v[0]); \
        \
        in_v = _mm256_maskload_ps(input_temp + 10*WIDTH + 10, mask1); \
        \
        sum_v[0] = _mm256_fmadd_ps(_mm256_load_ps(w0 + 120), in_v, sum_v[0]);

        MY_CONVOLVE(input_temp0, weight_temp0);
        MY_CONVOLVE(input_temp1, weight_temp1);
        MY_CONVOLVE(input_temp2, weight_temp2);

        _mm256_store_ps(sum_temp0, sum_v[0]);
        float sum0 = 0;
        for (int i = 0; i < 8; ++i) {
          sum0 += sum_temp0[i];
        }

        output[(out_channel * WOUT + output_row) * WOUT + output_col] = sum0;
      } // for each output col
    } // for each output row
  } // for each out channel
}

extern unsigned long long conv_cycles_of_this_batch[1024*16], transpose_cycle, pool_cycle;

// JSP: AlexNet each group of conv3-5
// Input: 256 x 13 x 13 => 1.3 KB per channel, 338 KB total
// Output: 384 x 13 x 13 => 1.3 KB per channel, 507 KB total
// Weight: 384 x 256 x 3 x 3 => 72B per channel pair, 18 KB per output channel, 27 KB per input channel, 6.8 MB total
//         No matter what we do, there's no reuse on weight across different channels (only reuse is within a channel pair)
// FLOPS: 2 x 384 x 256 x 13 x 13 x 3 x 3 = 299 MFLOPS

// blocking by 16 input channels

static /*inline */ void __attribute__((noinline)) sconv345_ver2(
    // input features
    const float *input, int in_channels,
    // weights
    const int *blockptr, const int *kidx, const float *values,
    // bias (for the case when bias is fused with convolution)
    const float *bias,
    // output features
    float *output, int out_channels, float *input_scratch_global) // scratch: 6 KB per ic
{
  unsigned long long t = __rdtsc();

  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();

  const int WIDTH = 13;
  const int WOUT = 13;
  const int PAD = 1;
  const int K = 3;

  int nthread_groups = nthreads;
#ifdef __AVX512F__
  nthread_groups = NTILES;
#else
//  nthread_groups /= 2; // 1 group per core in Xeon
#endif
  assert(nthreads%nthread_groups == 0);
  int nthreads_per_group = nthreads/nthread_groups;
  int gid = tid/nthreads_per_group;
  int tid_in_group = tid%nthreads_per_group;

  static const int SCRATCH_SIZE_PER_IC = (WOUT*WOUT + 15)/16*16; // 176
  float *input_scratch = input_scratch_global + gid*COL_MAJOR_IC_BLOCK*K*K*SCRATCH_SIZE_PER_IC;

  float sum[WOUT][WOUT];

#if 1 // def __AVX512F__

#ifdef COL_MAJOR_OC_BLOCK
  int oc_block_per_thread = (out_channels/COL_MAJOR_OC_BLOCK + nthreads_per_group - 1)/nthreads_per_group;
  int oc_block_begin = std::min(oc_block_per_thread*tid_in_group, out_channels/COL_MAJOR_OC_BLOCK);
  int oc_block_end = std::min(oc_block_begin + oc_block_per_thread, out_channels/COL_MAJOR_OC_BLOCK);

  for (int oc_block = oc_block_begin; oc_block < oc_block_end; ++oc_block) {
#else
    int oc_per_thread = (out_channels + nthreads_per_group - 1)/nthreads_per_group;
    int oc_begin = std::min(oc_per_thread*tid_in_group, out_channels);
    int oc_end = std::min(oc_begin + oc_per_thread, out_channels);
#endif

    int ic_per_thread = (COL_MAJOR_IC_BLOCK + nthreads_per_group - 1)/nthreads_per_group;
    int ic_begin = std::min(ic_per_thread*tid_in_group, COL_MAJOR_IC_BLOCK);
    int ic_end = std::min(ic_begin + ic_per_thread, COL_MAJOR_IC_BLOCK);

    for (int ic = ic_begin; ic < ic_end; ++ic) {
      __m512 v = _mm512_loadu_ps(input + (ic*(WIDTH + PAD) + 1)*(WIDTH + PAD));
      _mm512_mask_storeu_ps(input_scratch + ic*K*K*SCRATCH_SIZE_PER_IC + 1*WOUT, 0x1fff, v);
      _mm512_mask_compressstoreu_ps(input_scratch + (ic*K*K + 1)*SCRATCH_SIZE_PER_IC + 1*WOUT, 0x3ffe, v);
      _mm512_mask_compressstoreu_ps(input_scratch + (ic*K*K + 2)*SCRATCH_SIZE_PER_IC + 1*WOUT, 0x7ffc, v);
      _mm512_mask_storeu_ps(input_scratch + (ic*K + 1)*K*SCRATCH_SIZE_PER_IC + 0*WOUT, 0x1fff, v);
      _mm512_mask_compressstoreu_ps(input_scratch + ((ic*K + 1)*K + 1)*SCRATCH_SIZE_PER_IC + 0*WOUT, 0x3ffe, v);
      _mm512_mask_compressstoreu_ps(input_scratch + ((ic*K + 1)*K + 2)*SCRATCH_SIZE_PER_IC + 0*WOUT, 0x7ffc, v);

      for (int h = 2; h < WOUT; ++h) {
        v = _mm512_loadu_ps(input + (ic*(WIDTH + PAD) + h)*(WIDTH + PAD));
        _mm512_mask_storeu_ps(input_scratch + ic*K*K*SCRATCH_SIZE_PER_IC + h*WOUT, 0x1fff, v);
        _mm512_mask_compressstoreu_ps(input_scratch + (ic*K*K + 1)*SCRATCH_SIZE_PER_IC + h*WOUT, 0x3ffe, v);
        _mm512_mask_compressstoreu_ps(input_scratch + (ic*K*K + 2)*SCRATCH_SIZE_PER_IC + h*WOUT, 0x7ffc, v);
        _mm512_mask_storeu_ps(input_scratch + (ic*K + 1)*K*SCRATCH_SIZE_PER_IC + (h - 1)*WOUT, 0x1fff, v);
        _mm512_mask_compressstoreu_ps(input_scratch + ((ic*K + 1)*K + 1)*SCRATCH_SIZE_PER_IC + (h - 1)*WOUT, 0x3ffe, v);
        _mm512_mask_compressstoreu_ps(input_scratch + ((ic*K + 1)*K + 2)*SCRATCH_SIZE_PER_IC + (h - 1)*WOUT, 0x7ffc, v);
        _mm512_mask_storeu_ps(input_scratch + (ic*K + 2)*K*SCRATCH_SIZE_PER_IC + (h - 2)*WOUT, 0x1fff, v);
        _mm512_mask_compressstoreu_ps(input_scratch + ((ic*K + 2)*K + 1)*SCRATCH_SIZE_PER_IC + (h - 2)*WOUT, 0x3ffe, v);
        _mm512_mask_compressstoreu_ps(input_scratch + ((ic*K + 2)*K + 2)*SCRATCH_SIZE_PER_IC + (h - 2)*WOUT, 0x7ffc, v);
      }

      v = _mm512_loadu_ps(input + (ic*(WIDTH + PAD) + WOUT)*(WIDTH + PAD));
      _mm512_mask_storeu_ps(input_scratch + (ic*K + 1)*K*SCRATCH_SIZE_PER_IC + (WOUT - 1)*WOUT, 0x1fff, v);
      _mm512_mask_compressstoreu_ps(input_scratch + ((ic*K + 1)*K + 1)*SCRATCH_SIZE_PER_IC + (WOUT - 1)*WOUT, 0x3ffe, v);
      _mm512_mask_compressstoreu_ps(input_scratch + ((ic*K + 1)*K + 2)*SCRATCH_SIZE_PER_IC + (WOUT - 1)*WOUT, 0x7ffc, v);
      _mm512_mask_storeu_ps(input_scratch + (ic*K + 2)*K*SCRATCH_SIZE_PER_IC + (WOUT - 2)*WOUT, 0x1fff, v);
      _mm512_mask_compressstoreu_ps(input_scratch + ((ic*K + 2)*K + 1)*SCRATCH_SIZE_PER_IC + (WOUT - 2)*WOUT, 0x3ffe, v);
      _mm512_mask_compressstoreu_ps(input_scratch + ((ic*K + 2)*K + 2)*SCRATCH_SIZE_PER_IC + (WOUT - 2)*WOUT, 0x7ffc, v);
    }

    if (nthread_groups != nthreads) barriers[gid]->wait(tid_in_group);

    __m512 sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10;

#ifdef COL_MAJOR_OC_BLOCK
    for (int oc = oc_block*COL_MAJOR_OC_BLOCK; oc < (oc_block + 1)*COL_MAJOR_OC_BLOCK; ++oc) {
#else
    for (int oc = oc_begin; oc < oc_end; ++oc) {
#endif
      __m512 bias_v = _mm512_set1_ps(bias[oc]);

      sum0 = bias_v;
      sum1 = bias_v;
      sum2 = bias_v;
      sum3 = bias_v;
      sum4 = bias_v;
      sum5 = bias_v;
      sum6 = bias_v;
      sum7 = bias_v;
      sum8 = bias_v;
      sum9 = bias_v;
      sum10 = bias_v;

      for (int b = blockptr[oc]; b < blockptr[oc + 1]; ++b) {
        int off = kidx[b];
        __m512 c = _mm512_set1_ps(values[b]);

        sum0 = _mm512_fmadd_ps(c, _mm512_load_ps(input_scratch + off + 0*16), sum0);
        sum1 = _mm512_fmadd_ps(c, _mm512_load_ps(input_scratch + off + 1*16), sum1);
        sum2 = _mm512_fmadd_ps(c, _mm512_load_ps(input_scratch + off + 2*16), sum2);
        sum3 = _mm512_fmadd_ps(c, _mm512_load_ps(input_scratch + off + 3*16), sum3);
        sum4 = _mm512_fmadd_ps(c, _mm512_load_ps(input_scratch + off + 4*16), sum4);
        sum5 = _mm512_fmadd_ps(c, _mm512_load_ps(input_scratch + off + 5*16), sum5);
        sum6 = _mm512_fmadd_ps(c, _mm512_load_ps(input_scratch + off + 6*16), sum6);
        sum7 = _mm512_fmadd_ps(c, _mm512_load_ps(input_scratch + off + 7*16), sum7);
        sum8 = _mm512_fmadd_ps(c, _mm512_load_ps(input_scratch + off + 8*16), sum8);
        sum9 = _mm512_fmadd_ps(c, _mm512_load_ps(input_scratch + off + 9*16), sum9);
        sum10 = _mm512_fmadd_ps(c, _mm512_load_ps(input_scratch + off + 10*16), sum10);
      }

      _mm512_storeu_ps(output + oc*WOUT*WOUT + 0*16, sum0);
      _mm512_storeu_ps(output + oc*WOUT*WOUT + 1*16, sum1);
      _mm512_storeu_ps(output + oc*WOUT*WOUT + 2*16, sum2);
      _mm512_storeu_ps(output + oc*WOUT*WOUT + 3*16, sum3);
      _mm512_storeu_ps(output + oc*WOUT*WOUT + 4*16, sum4);
      _mm512_storeu_ps(output + oc*WOUT*WOUT + 5*16, sum5);
      _mm512_storeu_ps(output + oc*WOUT*WOUT + 6*16, sum6);
      _mm512_storeu_ps(output + oc*WOUT*WOUT + 7*16, sum7);
      _mm512_storeu_ps(output + oc*WOUT*WOUT + 8*16, sum8);
      _mm512_storeu_ps(output + oc*WOUT*WOUT + 9*16, sum9);
      _mm512_mask_storeu_ps(output + oc*WOUT*WOUT + 10*16, 0x01ff, sum10);
    }

    for (int ic_block = 1; ic_block < in_channels/COL_MAJOR_IC_BLOCK; ++ic_block) {

      if (nthread_groups != nthreads) barriers[gid]->wait(tid_in_group);

      for (int ic = ic_begin; ic < ic_end; ++ic) {
        __m512 v = _mm512_loadu_ps(input + ((ic + ic_block*COL_MAJOR_IC_BLOCK)*(WIDTH + PAD) + 1)*(WIDTH + PAD));
        _mm512_mask_storeu_ps(input_scratch + ic*K*K*SCRATCH_SIZE_PER_IC + 1*WOUT, 0x1fff, v);
        _mm512_mask_compressstoreu_ps(input_scratch + (ic*K*K + 1)*SCRATCH_SIZE_PER_IC + 1*WOUT, 0x3ffe, v);
        _mm512_mask_compressstoreu_ps(input_scratch + (ic*K*K + 2)*SCRATCH_SIZE_PER_IC + 1*WOUT, 0x7ffc, v);
        _mm512_mask_storeu_ps(input_scratch + (ic*K + 1)*K*SCRATCH_SIZE_PER_IC + 0*WOUT, 0x1fff, v);
        _mm512_mask_compressstoreu_ps(input_scratch + ((ic*K + 1)*K + 1)*SCRATCH_SIZE_PER_IC + 0*WOUT, 0x3ffe, v);
        _mm512_mask_compressstoreu_ps(input_scratch + ((ic*K + 1)*K + 2)*SCRATCH_SIZE_PER_IC + 0*WOUT, 0x7ffc, v);

        for (int h = 2; h < WOUT; ++h) {
          v = _mm512_loadu_ps(input + ((ic + ic_block*COL_MAJOR_IC_BLOCK)*(WIDTH + PAD) + h)*(WIDTH + PAD));
          _mm512_mask_storeu_ps(input_scratch + ic*K*K*SCRATCH_SIZE_PER_IC + h*WOUT, 0x1fff, v);
          _mm512_mask_compressstoreu_ps(input_scratch + (ic*K*K + 1)*SCRATCH_SIZE_PER_IC + h*WOUT, 0x3ffe, v);
          _mm512_mask_compressstoreu_ps(input_scratch + (ic*K*K + 2)*SCRATCH_SIZE_PER_IC + h*WOUT, 0x7ffc, v);
          _mm512_mask_storeu_ps(input_scratch + (ic*K + 1)*K*SCRATCH_SIZE_PER_IC + (h - 1)*WOUT, 0x1fff, v);
          _mm512_mask_compressstoreu_ps(input_scratch + ((ic*K + 1)*K + 1)*SCRATCH_SIZE_PER_IC + (h - 1)*WOUT, 0x3ffe, v);
          _mm512_mask_compressstoreu_ps(input_scratch + ((ic*K + 1)*K + 2)*SCRATCH_SIZE_PER_IC + (h - 1)*WOUT, 0x7ffc, v);
          _mm512_mask_storeu_ps(input_scratch + (ic*K + 2)*K*SCRATCH_SIZE_PER_IC + (h - 2)*WOUT, 0x1fff, v);
          _mm512_mask_compressstoreu_ps(input_scratch + ((ic*K + 2)*K + 1)*SCRATCH_SIZE_PER_IC + (h - 2)*WOUT, 0x3ffe, v);
          _mm512_mask_compressstoreu_ps(input_scratch + ((ic*K + 2)*K + 2)*SCRATCH_SIZE_PER_IC + (h - 2)*WOUT, 0x7ffc, v);
        }

        v = _mm512_loadu_ps(input + ((ic + ic_block*COL_MAJOR_IC_BLOCK)*(WIDTH + PAD) + WOUT)*(WIDTH + PAD));
        _mm512_mask_storeu_ps(input_scratch + (ic*K + 1)*K*SCRATCH_SIZE_PER_IC + (WOUT - 1)*WOUT, 0x1fff, v);
        _mm512_mask_compressstoreu_ps(input_scratch + ((ic*K + 1)*K + 1)*SCRATCH_SIZE_PER_IC + (WOUT - 1)*WOUT, 0x3ffe, v);
        _mm512_mask_compressstoreu_ps(input_scratch + ((ic*K + 1)*K + 2)*SCRATCH_SIZE_PER_IC + (WOUT - 1)*WOUT, 0x7ffc, v);
        _mm512_mask_storeu_ps(input_scratch + (ic*K + 2)*K*SCRATCH_SIZE_PER_IC + (WOUT - 2)*WOUT, 0x1fff, v);
        _mm512_mask_compressstoreu_ps(input_scratch + ((ic*K + 2)*K + 1)*SCRATCH_SIZE_PER_IC + (WOUT - 2)*WOUT, 0x3ffe, v);
        _mm512_mask_compressstoreu_ps(input_scratch + ((ic*K + 2)*K + 2)*SCRATCH_SIZE_PER_IC + (WOUT - 2)*WOUT, 0x7ffc, v);
      }

      if (nthread_groups != nthreads) barriers[gid]->wait(tid_in_group);

#ifdef COL_MAJOR_OC_BLOCK
      for (int oc = oc_block*COL_MAJOR_OC_BLOCK; oc < (oc_block + 1)*COL_MAJOR_OC_BLOCK; ++oc) {
#else
      for (int oc = oc_begin; oc < oc_end; ++oc) {
#endif
        sum0 = _mm512_loadu_ps(output + oc*WOUT*WOUT + 0*16);
        sum1 = _mm512_loadu_ps(output + oc*WOUT*WOUT + 1*16);
        sum2 = _mm512_loadu_ps(output + oc*WOUT*WOUT + 2*16);
        sum3 = _mm512_loadu_ps(output + oc*WOUT*WOUT + 3*16);
        sum4 = _mm512_loadu_ps(output + oc*WOUT*WOUT + 4*16);
        sum5 = _mm512_loadu_ps(output + oc*WOUT*WOUT + 5*16);
        sum6 = _mm512_loadu_ps(output + oc*WOUT*WOUT + 6*16);
        sum7 = _mm512_loadu_ps(output + oc*WOUT*WOUT + 7*16);
        sum8 = _mm512_loadu_ps(output + oc*WOUT*WOUT + 8*16);
        sum9 = _mm512_loadu_ps(output + oc*WOUT*WOUT + 9*16);
        sum10 = _mm512_loadu_ps(output + oc*WOUT*WOUT + 10*16);

        for (int b = blockptr[ic_block*out_channels + oc]; b < blockptr[ic_block*out_channels + oc + 1]; ++b) {
          int off = kidx[b];
          __m512 c = _mm512_set1_ps(values[b]);

          sum0 = _mm512_fmadd_ps(c, _mm512_load_ps(input_scratch + off + 0*16), sum0);
          sum1 = _mm512_fmadd_ps(c, _mm512_load_ps(input_scratch + off + 1*16), sum1);
          sum2 = _mm512_fmadd_ps(c, _mm512_load_ps(input_scratch + off + 2*16), sum2);
          sum3 = _mm512_fmadd_ps(c, _mm512_load_ps(input_scratch + off + 3*16), sum3);
          sum4 = _mm512_fmadd_ps(c, _mm512_load_ps(input_scratch + off + 4*16), sum4);
          sum5 = _mm512_fmadd_ps(c, _mm512_load_ps(input_scratch + off + 5*16), sum5);
          sum6 = _mm512_fmadd_ps(c, _mm512_load_ps(input_scratch + off + 6*16), sum6);
          sum7 = _mm512_fmadd_ps(c, _mm512_load_ps(input_scratch + off + 7*16), sum7);
          sum8 = _mm512_fmadd_ps(c, _mm512_load_ps(input_scratch + off + 8*16), sum8);
          sum9 = _mm512_fmadd_ps(c, _mm512_load_ps(input_scratch + off + 9*16), sum9);
          sum10 = _mm512_fmadd_ps(c, _mm512_load_ps(input_scratch + off + 10*16), sum10);
        }

        _mm512_storeu_ps(output + oc*WOUT*WOUT + 0*16, sum0);
        _mm512_storeu_ps(output + oc*WOUT*WOUT + 1*16, sum1);
        _mm512_storeu_ps(output + oc*WOUT*WOUT + 2*16, sum2);
        _mm512_storeu_ps(output + oc*WOUT*WOUT + 3*16, sum3);
        _mm512_storeu_ps(output + oc*WOUT*WOUT + 4*16, sum4);
        _mm512_storeu_ps(output + oc*WOUT*WOUT + 5*16, sum5);
        _mm512_storeu_ps(output + oc*WOUT*WOUT + 6*16, sum6);
        _mm512_storeu_ps(output + oc*WOUT*WOUT + 7*16, sum7);
        _mm512_storeu_ps(output + oc*WOUT*WOUT + 8*16, sum8);
        _mm512_storeu_ps(output + oc*WOUT*WOUT + 9*16, sum9);
        _mm512_mask_storeu_ps(output + oc*WOUT*WOUT + 10*16, 0x01ff, sum10);
      } // for each oc
    } // for each ic block
#ifdef COL_MAJOR_OC_BLOCK
  } // for each oc block
#endif
#else
  for (int oc = 0; oc < out_channels; ++oc) {

    for (int h = 0; h < WOUT; ++h) {
      for (int w = 0; w < WOUT; ++w) {
        sum[h][w] = bias[oc];
      }
    }

    for (int b = blockptr[oc]; b < blockptr[oc + 1]; ++b) {
      for (int h = 0; h < WOUT; ++h) {
        for (int w = 0; w < WOUT; ++w) {
          int k = kidx[b]/176%K + kidx[b]/176/K*(WIDTH + PAD); // /176%K + kidx[b]/176/K*(WIDTH + PAD);
          sum[h][w] += values[b]*input[k + h*(WIDTH + PAD) + w];
        }
      }
    }

    for (int h = 0; h < WOUT; ++h) {
      for (int w = 0; w < WOUT; ++w) {
        output[(oc*WOUT + h)*WOUT + w] = sum[h][w];
      }
    }
  }

  for (int ic = 1; ic < in_channels; ++ic) {
    for (int oc = 0; oc < out_channels; ++oc) {
      for (int h = 0; h < WOUT; ++h) {
        for (int w = 0; w < WOUT; ++w) {
          sum[h][w] = output[(oc*WOUT + h)*WOUT + w];
        }
      }

      for (int b = blockptr[ic*out_channels + oc]; b < blockptr[ic*out_channels + oc + 1]; ++b) {
        for (int h = 0; h < WOUT; ++h) {
          for (int w = 0; w < WOUT; ++w) {
            int k = kidx[b]/176%K + kidx[b]/176/K*(WIDTH + PAD); // /176%K + kidx[b]/176/K*(WIDTH + PAD);
            sum[h][w] += values[b]*input[k + (ic*(WIDTH + PAD) + h)*(WIDTH + PAD) + w];
          }
        }
      }

      for (int h = 0; h < WOUT; ++h) {
        for (int w = 0; w < WOUT; ++w) {
          output[(oc*WOUT + h)*WOUT + w] = sum[h][w];
        }
      }
    } // for each out channel
  } // for each input channel
#endif

  conv_cycles_of_this_batch[tid*16] += __rdtsc() - t;
}

/**
 * Direct sparse convolution optimized for 3-5 layers of AlexNet
 */
static inline void /*__attribute__((noinline))*/ sconv345(
    // input features
    const float *input,
    // weights
    const int *rowptr, const int *colidx, const float *values,
    const int **rowptr_blocked, const int **colidx_blocked, const float **values_blocked,
    int ncolblocks,
    // bias (for the case when bias is fused with convolution)
    const float *bias,
    // output features
    float *output,
    int out_channels,
    float *scratch) // scratch: 832B per OC_BLOCK
{
  unsigned long long t = __rdtsc();

  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();

  const int WIDTH = 13;
  const int WOUT = 13;
  const int PAD = 1;

  int nthread_groups = nthreads;
#ifdef __AVX512F__
  nthread_groups = NTILES;
#else
//  nthread_groups /= 2; // 1 group per core in Xeon
#endif
  assert(nthreads%nthread_groups == 0);
  int nthreads_per_group = nthreads/nthread_groups;
  int gid = tid/nthreads_per_group;
  int tid_in_group = tid%nthreads_per_group;

  int c_per_thread = (out_channels/OC_BLOCK + nthreads_per_group - 1)/nthreads_per_group;
  int c_begin = std::min(c_per_thread*tid_in_group, out_channels/OC_BLOCK);
  int c_end = std::min(c_begin + c_per_thread, out_channels/OC_BLOCK);

#ifndef __AVX512F__
  __declspec(aligned(64)) int mask_temp[8] = { -1, -1, -1, -1, -1, 0, 0, 0 };
  __m256i mask_v = _mm256_load_si256((__m256i *)mask_temp);
#endif

  for (int out_channel_begin = c_begin*OC_BLOCK; out_channel_begin < c_end*OC_BLOCK; out_channel_begin += OC_BLOCK) {
#ifdef __AVX512F__
    __m512 sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12;

    const int *rowptr = rowptr_blocked[0];
    const int *colidx = colidx_blocked[0];
    const float *values = values_blocked[0];

    for (int out_channel = out_channel_begin; out_channel < out_channel_begin + OC_BLOCK; ++out_channel) {
      __m512 bias_v = _mm512_set1_ps(bias[out_channel]);

      int hbegin = 0, hend = WOUT;

      sum0 = bias_v;
      sum1 = bias_v;
      sum2 = bias_v;
      sum3 = bias_v;
      sum4 = bias_v;
      sum5 = bias_v;
      sum6 = bias_v;
      sum7 = bias_v;
      sum8 = bias_v;
      sum9 = bias_v;
      sum10 = bias_v;
      sum11 = bias_v;
      sum12 = bias_v;

      int jbegin = rowptr[out_channel];
      int jend = rowptr[out_channel + 1];

#define W_PREFETCH_DISTANCE (1)

//      _mm_prefetch((const char *)(values + rowptr[out_channel + W_PREFETCH_DISTANCE]), _MM_HINT_T1);
//      _mm_prefetch((const char *)(values + rowptr[out_channel + W_PREFETCH_DISTANCE] + 16), _MM_HINT_T1);
//      _mm_prefetch((const char *)(colidx + rowptr[out_channel + W_PREFETCH_DISTANCE]), _MM_HINT_T1);
//      _mm_prefetch((const char *)(colidx + rowptr[out_channel + W_PREFETCH_DISTANCE] + 16), _MM_HINT_T1);

      for (int j = jbegin; j < jend; ++j) {
//#define HACK_WEIGHT
#ifdef HACK_WEIGHT
        __m512 c = _mm512_set1_ps(3);
        int off = j*16;
#else
        __m512 c = _mm512_set1_ps(values[j]);
        int off = colidx[j];
#endif

//#define HACK_ALIGN
#if defined(HACK_ALIGN) && defined(HACK_WEIGHT)
        sum0 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 0*(16)), sum0);
        sum1 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 1*(16)), sum1);
        sum2 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 2*(16)), sum2);
        sum3 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 3*(16)), sum3);
        sum4 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 4*(16)), sum4);
        sum5 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 5*(16)), sum5);
        sum6 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 6*(16)), sum6);
        sum7 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 7*(16)), sum7);
        sum8 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 8*(16)), sum8);
        sum9 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 9*(16)), sum9);
        sum10 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 10*(16)), sum10);
        sum11 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 11*(16)), sum11);
        sum12 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 12*(16)), sum12);
#else
        sum0 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 0*(WIDTH + PAD)), sum0);
        sum1 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 1*(WIDTH + PAD)), sum1);
        sum2 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 2*(WIDTH + PAD)), sum2);
        sum3 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 3*(WIDTH + PAD)), sum3);
        sum4 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 4*(WIDTH + PAD)), sum4);
        sum5 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 5*(WIDTH + PAD)), sum5);
        sum6 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 6*(WIDTH + PAD)), sum6);
        sum7 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 7*(WIDTH + PAD)), sum7);
        sum8 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 8*(WIDTH + PAD)), sum8);
        sum9 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 9*(WIDTH + PAD)), sum9);
        sum10 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 10*(WIDTH + PAD)), sum10);
        sum11 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 11*(WIDTH + PAD)), sum11);
        sum12 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 12*(WIDTH + PAD)), sum12);
#endif
      }

//#define HACK_OUTPUT
#ifdef HACK_OUTPUT
      _mm512_store_ps(scratch + (0*WOUT + 0)*16, sum0);
      _mm512_store_ps(scratch + (0*WOUT + 1)*16, sum1);
      _mm512_store_ps(scratch + (0*WOUT + 2)*16, sum2);
      _mm512_store_ps(scratch + (0*WOUT + 3)*16, sum3);
      _mm512_store_ps(scratch + (0*WOUT + 4)*16, sum4);
      _mm512_store_ps(scratch + (0*WOUT + 5)*16, sum5);
      _mm512_store_ps(scratch + (0*WOUT + 6)*16, sum6);
      _mm512_store_ps(scratch + (0*WOUT + 7)*16, sum7);
      _mm512_store_ps(scratch + (0*WOUT + 8)*16, sum8);
      _mm512_store_ps(scratch + (0*WOUT + 9)*16, sum9);
      _mm512_store_ps(scratch + (0*WOUT + 10)*16, sum10);
      _mm512_store_ps(scratch + (0*WOUT + 11)*16, sum11);
      _mm512_store_ps(scratch + (0*WOUT + 12)*16, sum12);
#else
      _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 0)*16, sum0);
      _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 1)*16, sum1);
      _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 2)*16, sum2);
      _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 3)*16, sum3);
      _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 4)*16, sum4);
      _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 5)*16, sum5);
      _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 6)*16, sum6);
      _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 7)*16, sum7);
      _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 8)*16, sum8);
      _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 9)*16, sum9);
      _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 10)*16, sum10);
      _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 11)*16, sum11);
      _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 12)*16, sum12);
#endif

//#define PREFETCH_OUTPUT
#ifdef PREFETCH_OUTPUT
      _mm_prefetch((const char *)(output + out_channel*WOUT*WOUT), _MM_HINT_T0);
      _mm_prefetch((const char *)(output + out_channel*WOUT*WOUT + 16), _MM_HINT_T0);
      _mm_prefetch((const char *)(output + out_channel*WOUT*WOUT + 32), _MM_HINT_T0);
      _mm_prefetch((const char *)(output + out_channel*WOUT*WOUT + 48), _MM_HINT_T0);
      _mm_prefetch((const char *)(output + out_channel*WOUT*WOUT + 64), _MM_HINT_T0);
      _mm_prefetch((const char *)(output + out_channel*WOUT*WOUT + 80), _MM_HINT_T0);
      _mm_prefetch((const char *)(output + out_channel*WOUT*WOUT + 96), _MM_HINT_T0);
      _mm_prefetch((const char *)(output + out_channel*WOUT*WOUT + 112), _MM_HINT_T0);
      _mm_prefetch((const char *)(output + out_channel*WOUT*WOUT + 128), _MM_HINT_T0);
      _mm_prefetch((const char *)(output + out_channel*WOUT*WOUT + 144), _MM_HINT_T0);
      _mm_prefetch((const char *)(output + out_channel*WOUT*WOUT + 160), _MM_HINT_T0);
#endif
    }

    for (int b = 1; b < ncolblocks - 1; ++b) {
      rowptr = rowptr_blocked[b];
      colidx = colidx_blocked[b];
      values = values_blocked[b];

      for (int out_channel = out_channel_begin; out_channel < out_channel_begin + OC_BLOCK; ++out_channel) {
#ifdef HACK_OUTPUT
        sum0 = _mm512_load_ps(scratch + (0*WOUT + 0)*16);
        sum1 = _mm512_load_ps(scratch + (0*WOUT + 1)*16);
        sum2 = _mm512_load_ps(scratch + (0*WOUT + 2)*16);
        sum3 = _mm512_load_ps(scratch + (0*WOUT + 3)*16);
        sum4 = _mm512_load_ps(scratch + (0*WOUT + 4)*16);
        sum5 = _mm512_load_ps(scratch + (0*WOUT + 5)*16);
        sum6 = _mm512_load_ps(scratch + (0*WOUT + 6)*16);
        sum7 = _mm512_load_ps(scratch + (0*WOUT + 7)*16);
        sum8 = _mm512_load_ps(scratch + (0*WOUT + 8)*16);
        sum9 = _mm512_load_ps(scratch + (0*WOUT + 9)*16);
        sum10 = _mm512_load_ps(scratch + (0*WOUT + 10)*16);
        sum11 = _mm512_load_ps(scratch + (0*WOUT + 11)*16);
        sum12 = _mm512_load_ps(scratch + (0*WOUT + 12)*16);
#else
        sum0 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 0)*16);
        sum1 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 1)*16);
        sum2 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 2)*16);
        sum3 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 3)*16);
        sum4 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 4)*16);
        sum5 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 5)*16);
        sum6 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 6)*16);
        sum7 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 7)*16);
        sum8 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 8)*16);
        sum9 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 9)*16);
        sum10 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 10)*16);
        sum11 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 11)*16);
        sum12 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 12)*16);
#endif

        int jbegin = rowptr[out_channel];
        int jend = rowptr[out_channel + 1];

//        _mm_prefetch((const char *)(values + rowptr[out_channel + W_PREFETCH_DISTANCE]), _MM_HINT_T1);
//        _mm_prefetch((const char *)(values + rowptr[out_channel + W_PREFETCH_DISTANCE] + 16), _MM_HINT_T1);
//        _mm_prefetch((const char *)(colidx + rowptr[out_channel + W_PREFETCH_DISTANCE]), _MM_HINT_T1);
//        _mm_prefetch((const char *)(colidx + rowptr[out_channel + W_PREFETCH_DISTANCE] + 16), _MM_HINT_T1);

        for (int j = jbegin; j < jend; ++j) {
#ifdef HACK_WEIGHT
          __m512 c = _mm512_set1_ps(3);
          int off = j*16;
#else
          __m512 c = _mm512_set1_ps(values[j]);
          int off = colidx[j];
#endif

#if defined(HACK_ALIGN) && defined(HACK_WEIGHT)
          sum0 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 0*(16)), sum0);
          sum1 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 1*(16)), sum1);
          sum2 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 2*(16)), sum2);
          sum3 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 3*(16)), sum3);
          sum4 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 4*(16)), sum4);
          sum5 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 5*(16)), sum5);
          sum6 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 6*(16)), sum6);
          sum7 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 7*(16)), sum7);
          sum8 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 8*(16)), sum8);
          sum9 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 9*(16)), sum9);
          sum10 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 10*(16)), sum10);
          sum11 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 11*(16)), sum11);
          sum12 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 12*(16)), sum12);
#else
          sum0 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 0*(WIDTH + PAD)), sum0);
          sum1 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 1*(WIDTH + PAD)), sum1);
          sum2 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 2*(WIDTH + PAD)), sum2);
          sum3 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 3*(WIDTH + PAD)), sum3);
          sum4 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 4*(WIDTH + PAD)), sum4);
          sum5 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 5*(WIDTH + PAD)), sum5);
          sum6 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 6*(WIDTH + PAD)), sum6);
          sum7 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 7*(WIDTH + PAD)), sum7);
          sum8 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 8*(WIDTH + PAD)), sum8);
          sum9 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 9*(WIDTH + PAD)), sum9);
          sum10 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 10*(WIDTH + PAD)), sum10);
          sum11 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 11*(WIDTH + PAD)), sum11);
          sum12 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 12*(WIDTH + PAD)), sum12);
#endif
        }

#ifdef HACK_OUTPUT
        _mm512_store_ps(scratch + (0*WOUT + 0)*16, sum0);
        _mm512_store_ps(scratch + (0*WOUT + 1)*16, sum1);
        _mm512_store_ps(scratch + (0*WOUT + 2)*16, sum2);
        _mm512_store_ps(scratch + (0*WOUT + 3)*16, sum3);
        _mm512_store_ps(scratch + (0*WOUT + 4)*16, sum4);
        _mm512_store_ps(scratch + (0*WOUT + 5)*16, sum5);
        _mm512_store_ps(scratch + (0*WOUT + 6)*16, sum6);
        _mm512_store_ps(scratch + (0*WOUT + 7)*16, sum7);
        _mm512_store_ps(scratch + (0*WOUT + 8)*16, sum8);
        _mm512_store_ps(scratch + (0*WOUT + 9)*16, sum9);
        _mm512_store_ps(scratch + (0*WOUT + 10)*16, sum10);
        _mm512_store_ps(scratch + (0*WOUT + 11)*16, sum11);
        _mm512_store_ps(scratch + (0*WOUT + 12)*16, sum12);
#else
        _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 0)*16, sum0);
        _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 1)*16, sum1);
        _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 2)*16, sum2);
        _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 3)*16, sum3);
        _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 4)*16, sum4);
        _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 5)*16, sum5);
        _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 6)*16, sum6);
        _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 7)*16, sum7);
        _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 8)*16, sum8);
        _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 9)*16, sum9);
        _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 10)*16, sum10);
        _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 11)*16, sum11);
        _mm512_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 12)*16, sum12);
#endif
      } // for each out channel
    } // for each col block

    rowptr = rowptr_blocked[ncolblocks - 1];
    colidx = colidx_blocked[ncolblocks - 1];
    values = values_blocked[ncolblocks - 1];

    for (int out_channel = out_channel_begin; out_channel < out_channel_begin + OC_BLOCK; ++out_channel) {
#ifdef HACK_OUTPUT
      sum0 = _mm512_load_ps(scratch + (0*WOUT + 0)*16);
      sum1 = _mm512_load_ps(scratch + (0*WOUT + 1)*16);
      sum2 = _mm512_load_ps(scratch + (0*WOUT + 2)*16);
      sum3 = _mm512_load_ps(scratch + (0*WOUT + 3)*16);
      sum4 = _mm512_load_ps(scratch + (0*WOUT + 4)*16);
      sum5 = _mm512_load_ps(scratch + (0*WOUT + 5)*16);
      sum6 = _mm512_load_ps(scratch + (0*WOUT + 6)*16);
      sum7 = _mm512_load_ps(scratch + (0*WOUT + 7)*16);
      sum8 = _mm512_load_ps(scratch + (0*WOUT + 8)*16);
      sum9 = _mm512_load_ps(scratch + (0*WOUT + 9)*16);
      sum10 = _mm512_load_ps(scratch + (0*WOUT + 10)*16);
      sum11 = _mm512_load_ps(scratch + (0*WOUT + 11)*16);
      sum12 = _mm512_load_ps(scratch + (0*WOUT + 12)*16);
#else
      sum0 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 0)*16);
      sum1 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 1)*16);
      sum2 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 2)*16);
      sum3 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 3)*16);
      sum4 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 4)*16);
      sum5 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 5)*16);
      sum6 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 6)*16);
      sum7 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 7)*16);
      sum8 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 8)*16);
      sum9 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 9)*16);
      sum10 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 10)*16);
      sum11 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 11)*16);
      sum12 = _mm512_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + 12)*16);
#endif

      int jbegin = rowptr[out_channel];
      int jend = rowptr[out_channel + 1];

//      _mm_prefetch((const char *)(values + rowptr[out_channel + W_PREFETCH_DISTANCE]), _MM_HINT_T1);
//      _mm_prefetch((const char *)(values + rowptr[out_channel + W_PREFETCH_DISTANCE] + 16), _MM_HINT_T1);
//      _mm_prefetch((const char *)(colidx + rowptr[out_channel + W_PREFETCH_DISTANCE]), _MM_HINT_T1);
//      _mm_prefetch((const char *)(colidx + rowptr[out_channel + W_PREFETCH_DISTANCE] + 16), _MM_HINT_T1);

      for (int j = jbegin; j < jend; ++j) {
#ifdef HACK_WEIGHT
        __m512 c = _mm512_set1_ps(3);
        int off = j*16;
#else
        __m512 c = _mm512_set1_ps(values[j]);
        int off = colidx[j];
#endif

#if defined(HACK_ALIGN) && defined(HACK_WEIGHT)
        sum0 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 0*(16)), sum0);
        sum1 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 1*(16)), sum1);
        sum2 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 2*(16)), sum2);
        sum3 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 3*(16)), sum3);
        sum4 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 4*(16)), sum4);
        sum5 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 5*(16)), sum5);
        sum6 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 6*(16)), sum6);
        sum7 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 7*(16)), sum7);
        sum8 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 8*(16)), sum8);
        sum9 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 9*(16)), sum9);
        sum10 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 10*(16)), sum10);
        sum11 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 11*(16)), sum11);
        sum12 = _mm512_fmadd_ps(c, _mm512_load_ps(input + off + 12*(16)), sum12);
#else
        sum0 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 0*(WIDTH + PAD)), sum0);
        sum1 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 1*(WIDTH + PAD)), sum1);
        sum2 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 2*(WIDTH + PAD)), sum2);
        sum3 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 3*(WIDTH + PAD)), sum3);
        sum4 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 4*(WIDTH + PAD)), sum4);
        sum5 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 5*(WIDTH + PAD)), sum5);
        sum6 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 6*(WIDTH + PAD)), sum6);
        sum7 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 7*(WIDTH + PAD)), sum7);
        sum8 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 8*(WIDTH + PAD)), sum8);
        sum9 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 9*(WIDTH + PAD)), sum9);
        sum10 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 10*(WIDTH + PAD)), sum10);
        sum11 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 11*(WIDTH + PAD)), sum11);
        sum12 = _mm512_fmadd_ps(c, _mm512_loadu_ps(input + off + 12*(WIDTH + PAD)), sum12);
#endif
      }

#ifdef HACK_OUTPUT
      _mm512_mask_storeu_ps(output + (0*WOUT + 0)*WOUT, 0x1fff, sum0);
      _mm512_mask_storeu_ps(output + (0*WOUT + 1)*WOUT, 0x1fff, sum1);
      _mm512_mask_storeu_ps(output + (0*WOUT + 2)*WOUT, 0x1fff, sum2);
      _mm512_mask_storeu_ps(output + (0*WOUT + 3)*WOUT, 0x1fff, sum3);
      _mm512_mask_storeu_ps(output + (0*WOUT + 4)*WOUT, 0x1fff, sum4);
      _mm512_mask_storeu_ps(output + (0*WOUT + 5)*WOUT, 0x1fff, sum5);
      _mm512_mask_storeu_ps(output + (0*WOUT + 6)*WOUT, 0x1fff, sum6);
      _mm512_mask_storeu_ps(output + (0*WOUT + 7)*WOUT, 0x1fff, sum7);
      _mm512_mask_storeu_ps(output + (0*WOUT + 8)*WOUT, 0x1fff, sum8);
      _mm512_mask_storeu_ps(output + (0*WOUT + 9)*WOUT, 0x1fff, sum9);
      _mm512_mask_storeu_ps(output + (0*WOUT + 10)*WOUT, 0x1fff, sum10);
      _mm512_mask_storeu_ps(output + (0*WOUT + 11)*WOUT, 0x1fff, sum11);
      _mm512_mask_storeu_ps(output + (0*WOUT + 12)*WOUT, 0x1fff, sum12);
#else
      _mm512_mask_storeu_ps(output + (out_channel*WOUT + 0)*WOUT, 0x1fff, sum0);
      _mm512_mask_storeu_ps(output + (out_channel*WOUT + 1)*WOUT, 0x1fff, sum1);
      _mm512_mask_storeu_ps(output + (out_channel*WOUT + 2)*WOUT, 0x1fff, sum2);
      _mm512_mask_storeu_ps(output + (out_channel*WOUT + 3)*WOUT, 0x1fff, sum3);
      _mm512_mask_storeu_ps(output + (out_channel*WOUT + 4)*WOUT, 0x1fff, sum4);
      _mm512_mask_storeu_ps(output + (out_channel*WOUT + 5)*WOUT, 0x1fff, sum5);
      _mm512_mask_storeu_ps(output + (out_channel*WOUT + 6)*WOUT, 0x1fff, sum6);
      _mm512_mask_storeu_ps(output + (out_channel*WOUT + 7)*WOUT, 0x1fff, sum7);
      _mm512_mask_storeu_ps(output + (out_channel*WOUT + 8)*WOUT, 0x1fff, sum8);
      _mm512_mask_storeu_ps(output + (out_channel*WOUT + 9)*WOUT, 0x1fff, sum9);
      _mm512_mask_storeu_ps(output + (out_channel*WOUT + 10)*WOUT, 0x1fff, sum10);
      _mm512_mask_storeu_ps(output + (out_channel*WOUT + 11)*WOUT, 0x1fff, sum11);
      _mm512_mask_storeu_ps(output + (out_channel*WOUT + 12)*WOUT, 0x1fff, sum12);
#endif
    }
#else
    __m256 sum[(WOUT + 1)/2][2]; // [7][2]
    __m256 w_v;
    int off;

    const int *rowptr = rowptr_blocked[0];
    const int *colidx = colidx_blocked[0];
    const float *values = values_blocked[0];

    for (int out_channel = out_channel_begin; out_channel < out_channel_begin + OC_BLOCK; ++out_channel) {
      __m256 bias_v = _mm256_set1_ps(bias[out_channel]);

      // Upper half of images
      int hbegin = 0, hend = (WOUT + 1)/2;

#pragma unroll(7) // FIXME - compiler refuses to unroll this. Let's manually unroll it.
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin][0] = bias_v;
        sum[h - hbegin][1] = bias_v;
      }

      int jbegin = rowptr[out_channel];
      int jend = rowptr[out_channel + 1];

      for (int j = jbegin; j < jend; ++j) {
        w_v = _mm256_set1_ps(values[j]);
        off = colidx[j];

#pragma unroll(7)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
          sum[h - hbegin][1] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
        }
      }

#pragma unroll(7)
      for (int h = hbegin; h < hend; ++h) {
        _mm256_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + h)*16, sum[h - hbegin][0]);
        _mm256_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + h)*16 + 8, sum[h - hbegin][1]);
      }

      // Lower half of images
      hbegin = (WOUT + 1)/2; hend = WOUT;

#pragma unroll(6)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin][0] = bias_v;
        sum[h - hbegin][1] = bias_v;
      }

      for (int j = jbegin; j < jend; ++j) {
        w_v = _mm256_set1_ps(values[j]);
        off = colidx[j];

#pragma unroll(6)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
          sum[h - hbegin][1] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
        }
      }

#pragma unroll(6)
      for (int h = hbegin; h < hend; ++h) {
        _mm256_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + h)*16, sum[h - hbegin][0]);
        _mm256_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + h)*16 + 8, sum[h - hbegin][1]);
      }
    }

    for (int b = 1; b < ncolblocks - 1; ++b) {
      rowptr = rowptr_blocked[b];
      colidx = colidx_blocked[b];
      values = values_blocked[b];

      for (int out_channel = out_channel_begin; out_channel < out_channel_begin + OC_BLOCK; ++out_channel) {

        // Upper half of images
        int hbegin = 0, hend = (WOUT + 1)/2;

#pragma unroll(7)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + h)*16);
          sum[h - hbegin][1] = _mm256_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + h)*16 + 8);
        }

        int jbegin = rowptr[out_channel];
        int jend = rowptr[out_channel + 1];

        for (int j = jbegin; j < jend; ++j) {
          w_v = _mm256_set1_ps(values[j]);
          off = colidx[j];

#pragma unroll(7)
          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin][0] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
            sum[h - hbegin][1] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
          }
        }

#pragma unroll(7)
        for (int h = hbegin; h < hend; ++h) {
          _mm256_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + h)*16, sum[h - hbegin][0]);
          _mm256_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + h)*16 + 8, sum[h - hbegin][1]);
        }

        // Lower half of images
        hbegin = (WOUT + 1)/2; hend = WOUT;

#pragma unroll(6)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + h)*16);
          sum[h - hbegin][1] = _mm256_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + h)*16 + 8);
        }

        for (int j = jbegin; j < jend; ++j) {
          w_v = _mm256_set1_ps(values[j]);
          off = colidx[j];

#pragma unroll(6)
          for (int h = hbegin; h < hend; ++h) {
            sum[h - hbegin][0] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
            sum[h - hbegin][1] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
          }
        }

#pragma unroll(6)
        for (int h = hbegin; h < hend; ++h) {
          _mm256_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + h)*16, sum[h - hbegin][0]);
          _mm256_store_ps(scratch + ((out_channel - out_channel_begin)*WOUT + h)*16 + 8, sum[h - hbegin][1]);
        }
      }
    } // for each col block

    rowptr = rowptr_blocked[ncolblocks - 1];
    colidx = colidx_blocked[ncolblocks - 1];
    values = values_blocked[ncolblocks - 1];

    for (int out_channel = out_channel_begin; out_channel < out_channel_begin + OC_BLOCK; ++out_channel) {

      // Upper half of images
      int hbegin = 0, hend = (WOUT + 1)/2;

#pragma unroll(7)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin][0] = _mm256_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + h)*16);
        sum[h - hbegin][1] = _mm256_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + h)*16 + 8);
      }

      int jbegin = rowptr[out_channel];
      int jend = rowptr[out_channel + 1];

      for (int j = jbegin; j < jend; ++j) {
        w_v = _mm256_set1_ps(values[j]);
        off = colidx[j];

#pragma unroll(7)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
          sum[h - hbegin][1] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
        }
      }

#pragma unroll(7)
      for (int h = hbegin; h < hend; ++h) {
        _mm256_storeu_ps(output + (out_channel*WOUT + h)*WOUT, sum[h - hbegin][0]);
        _mm256_maskstore_ps(output + (out_channel*WOUT + h)*WOUT + 8, mask_v, sum[h - hbegin][1]);
      }

      // Lower half of images
      hbegin = (WOUT + 1)/2; hend = WOUT;

#pragma unroll(6)
      for (int h = hbegin; h < hend; ++h) {
        sum[h - hbegin][0] = _mm256_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + h)*16);
        sum[h - hbegin][1] = _mm256_load_ps(scratch + ((out_channel - out_channel_begin)*WOUT + h)*16 + 8);
      }

      for (int j = jbegin; j < jend; ++j) {
        w_v = _mm256_set1_ps(values[j]);
        off = colidx[j];

#pragma unroll(6)
        for (int h = hbegin; h < hend; ++h) {
          sum[h - hbegin][0] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
          sum[h - hbegin][1] = _mm256_fmadd_ps(w_v, _mm256_loadu_ps(input + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
        }
      }

#pragma unroll(6)
      for (int h = hbegin; h < hend; ++h) {
        _mm256_storeu_ps(output + (out_channel*WOUT + h)*WOUT, sum[h - hbegin][0]);
        _mm256_maskstore_ps(output + (out_channel*WOUT + h)*WOUT + 8, mask_v, sum[h - hbegin][1]);
      }
    }
#endif
  }

  conv_cycles_of_this_batch[tid*16] += __rdtsc() - t;
}

#endif /* SRC_CAFFE_LAYERS_CONV_HPP_ */
