#include <vector>
#include <omp.h>
#include <immintrin.h>

#include "caffe/layers/conv_relu_pool_lrn_layer.hpp"

unsigned long long conv_cycles_of_this_batch[1024*16], transpose_cycle = 0, pool_cycle = 0;
std::map<std::string, unsigned long long> total_conv_cycles;
std::map<std::string, double> total_conv_flops;
int total_files = 0;

double get_cpu_freq();

namespace caffe {

template <typename Dtype>
void ConvolutionReLUPoolLRNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
   BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, conv_top_);

   PoolingParameter pool_param = this->layer_param_.pooling_param();
   if (pool_param.global_pooling()) {
     CHECK(!(pool_param.has_kernel_size() ||
       pool_param.has_kernel_h() || pool_param.has_kernel_w()))
       << "With Global_pooling: true Filter size cannot specified";
   } else {
     CHECK(!pool_param.has_kernel_size() !=
       !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
       << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
     CHECK(pool_param.has_kernel_size() ||
       (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
       << "For non-square filters both kernel_h and kernel_w are required.";
   }
   CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
       && pool_param.has_pad_w())
       || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
       << "pad is pad OR pad_h and pad_w are required.";
   CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
       && pool_param.has_stride_w())
       || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
       << "Stride is stride OR stride_h and stride_w are required.";
   global_pooling_ = pool_param.global_pooling();
   if (global_pooling_) {
     kernel_h_ = bottom[0]->height();
     kernel_w_ = bottom[0]->width();
   } else {
     if (pool_param.has_kernel_size()) {
       kernel_h_ = kernel_w_ = pool_param.kernel_size();
     } else {
       kernel_h_ = pool_param.kernel_h();
       kernel_w_ = pool_param.kernel_w();
     }
   }
   CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
   CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
   if (!pool_param.has_pad_h()) {
     pad_h_ = pad_w_ = pool_param.pad();
   } else {
     pad_h_ = pool_param.pad_h();
     pad_w_ = pool_param.pad_w();
   }
   if (!pool_param.has_stride_h()) {
     stride_h_ = stride_w_ = pool_param.stride();
   } else {
     stride_h_ = pool_param.stride_h();
     stride_w_ = pool_param.stride_w();
   }
   if (global_pooling_) {
     CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
       << "With Global_pooling: true; only pad = 0 and stride = 1";
   }
   if (pad_h_ != 0 || pad_w_ != 0) {
     CHECK(this->layer_param_.pooling_param().pool()
         == PoolingParameter_PoolMethod_AVE
         || this->layer_param_.pooling_param().pool()
         == PoolingParameter_PoolMethod_MAX)
         << "Padding implemented only for average and max pooling.";
     CHECK_LT(pad_h_, kernel_h_);
     CHECK_LT(pad_w_, kernel_w_);
   }

   size_ = this->layer_param_.lrn_param().local_size();
   CHECK_EQ(size_ % 2, 1) << "LRN only supports odd values for local_size";
   pre_pad_ = (size_ - 1) / 2;
   alpha_ = this->layer_param_.lrn_param().alpha();
   beta_ = this->layer_param_.lrn_param().beta();
   k_ = this->layer_param_.lrn_param().k();
   if (this->layer_param_.lrn_param().norm_region() ==
       LRNParameter_NormRegion_WITHIN_CHANNEL) {
     // Set up split_layer_ to use inputs in the numerator and denominator.
     split_top_vec_.clear();
     split_top_vec_.push_back(&product_input_);
     split_top_vec_.push_back(&square_input_);
     LayerParameter split_param;
     split_layer_.reset(new SplitLayer<Dtype>(split_param));
     split_layer_->SetUp(pool_top_, split_top_vec_);
     // Set up square_layer_ to square the inputs.
     square_bottom_vec_.clear();
     square_top_vec_.clear();
     square_bottom_vec_.push_back(&square_input_);
     square_top_vec_.push_back(&square_output_);
     LayerParameter square_param;
     square_param.mutable_power_param()->set_power(Dtype(2));
     square_layer_.reset(new PowerLayer<Dtype>(square_param));
     square_layer_->SetUp(square_bottom_vec_, square_top_vec_);
     // Set up pool_layer_ to sum over square neighborhoods of the input.
     pool_top_vec_.clear();
     pool_top_vec_.push_back(&pool_output_);
     LayerParameter pool_param;
     pool_param.mutable_pooling_param()->set_pool(
         PoolingParameter_PoolMethod_AVE);
     pool_param.mutable_pooling_param()->set_pad(pre_pad_);
     pool_param.mutable_pooling_param()->set_kernel_size(size_);
     pool_layer_.reset(new PoolingLayer<Dtype>(pool_param));
     pool_layer_->SetUp(square_top_vec_, pool_top_vec_);
     // Set up power_layer_ to compute (1 + alpha_/N^2 s)^-beta_, where s is
     // the sum of a squared neighborhood (the output of pool_layer_).
     power_top_vec_.clear();
     power_top_vec_.push_back(&power_output_);
     LayerParameter power_param;
     power_param.mutable_power_param()->set_power(-beta_);
     power_param.mutable_power_param()->set_scale(alpha_);
     power_param.mutable_power_param()->set_shift(Dtype(1));
     power_layer_.reset(new PowerLayer<Dtype>(power_param));
     power_layer_->SetUp(pool_top_vec_, power_top_vec_);
     // Set up a product_layer_ to compute outputs by multiplying inputs by the
     // inverse demoninator computed by the power layer.
     product_bottom_vec_.clear();
     product_bottom_vec_.push_back(&product_input_);
     product_bottom_vec_.push_back(&power_output_);
     LayerParameter product_param;
     EltwiseParameter* eltwise_param = product_param.mutable_eltwise_param();
     eltwise_param->set_operation(EltwiseParameter_EltwiseOp_PROD);
     product_layer_.reset(new EltwiseLayer<Dtype>(product_param));
     product_layer_->SetUp(product_bottom_vec_, top);
   }
}

template <typename Dtype>
ConvolutionReLUPoolLRNLayer<Dtype>::~ConvolutionReLUPoolLRNLayer()
{
  free(scale_temp_);
  free(padded_square_);
}

template <typename Dtype>
void ConvolutionReLUPoolLRNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (conv_top_.empty()) {
    conv_top_.resize(bottom.size());
     for (int i = 0; i < conv_top_.size(); ++i) {
       conv_top_[i] = new Blob<Dtype>();
     }
  }
  BaseConvolutionLayer<Dtype>::Reshape(bottom, conv_top_);

  // Pooling
  if (pool_top_.empty()) {
    pool_top_.resize(bottom.size());
     for (int i = 0; i < pool_top_.size(); ++i) {
       pool_top_[i] = new Blob<Dtype>();
     }
  }

  CHECK_EQ(4, conv_top_[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
//  channels_ = middle_[0]->channels();
  height_ = conv_top_[0]->height();
  width_ = conv_top_[0]->width();
  if (global_pooling_) {
    kernel_h_ = conv_top_[0]->height();
    kernel_w_ = conv_top_[0]->width();
  }
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  if (pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
  }
  pool_top_[0]->Reshape(conv_top_[0]->num(), this->num_output_, pooled_height_,
      pooled_width_);
  if (pool_top_.size() > 1) {
    pool_top_[1]->ReshapeLike(*pool_top_[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && pool_top_.size() == 1) {
    max_idx_.Reshape(conv_top_[0]->num(), this->num_output_, pooled_height_,
        pooled_width_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(conv_top_[0]->num(), this->num_output_, pooled_height_,
      pooled_width_);
  }

  // LRN
  CHECK_EQ(4, pool_top_[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    top[0]->Reshape(this->num_, this->num_output_, pooled_height_, pooled_width_);
    scale_.Reshape(this->num_, this->num_output_, pooled_height_, pooled_width_);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    split_layer_->Reshape(pool_top_, split_top_vec_);
    square_layer_->Reshape(square_bottom_vec_, square_top_vec_);
    pool_layer_->Reshape(square_top_vec_, pool_top_vec_);
    power_layer_->Reshape(pool_top_vec_, power_top_vec_);
    product_layer_->Reshape(product_bottom_vec_, top);
    break;
  }
}

template <typename Dtype>
void ConvolutionReLUPoolLRNLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

double padding_time, im2col_time;

template<>
void ConvolutionReLUPoolLRNLayer<float>::Forward_cpu(const vector<Blob<float>*>& bottom,
      const vector<Blob<float>*>& top) {
  const float* weight = this->blobs_[0]->cpu_data();
  const float *bias = NULL;
  if (this->bias_term_) {
    bias = this->blobs_[1]->cpu_data();
  }

  float negative_slope = this->layer_param_.relu_param().negative_slope();
  assert(negative_slope == 0);

  float* pool_top = pool_top_[0]->mutable_cpu_data();
  int* mask = NULL;  // suppress warnings about uninitalized variables
  float* pool_top_mask = NULL;
  const bool use_pool_top_mask = pool_top_.size() > 1;
  if (use_pool_top_mask) {
    pool_top_mask = pool_top_[1]->mutable_cpu_data();
  }
  else {
    mask = max_idx_.mutable_cpu_data();
  }

  assert(this->layer_param_.lrn_param().norm_region() == LRNParameter_NormRegion_ACROSS_CHANNELS);
  if (!padded_square_) {
    posix_memalign(
        (void **)&padded_square_,
        4096,
        sizeof(float)*(omp_get_max_threads() * (this->num_output_ + size_ - 1) * width_));
  }
  float alpha_over_size = alpha_ / size_;
  if (!scale_temp_) {
    posix_memalign(
        (void **)&scale_temp_,
        4096,
        sizeof(float)*(omp_get_max_threads() * this->num_output_ * pooled_height_ * pooled_width_));
  }
  float* top_data = top[0]->mutable_cpu_data();

  double bias_time = 0;
  double pool_time = 0;
  double lrn_time = 0;
  padding_time = 0;
  im2col_time = 0;
  for (int i = 0; i < omp_get_max_threads(); ++i) {
    conv_cycles_of_this_batch[i*16] = 0;
  }
  transpose_cycle = 0;
  ::pool_cycle = 0;

  for (int i = 0; i < bottom.size(); ++i) {
    const float* bottom_data = bottom[i]->cpu_data();
    float* conv_top = conv_top_[i]->mutable_cpu_data();

#pragma omp parallel
    {
      int tid = omp_get_thread_num();

      float* padded_square_data = padded_square_ + tid * (this->num_output_ + size_ - 1) * width_;
      for (int j = 0; j < pre_pad_ * width_; ++j) {
        padded_square_data[j] = 0;
      }
      for (int j = (this->num_output_ + pre_pad_) * width_; j < (this->num_output_ + size_ - 1) * width_; ++j) {
        padded_square_data[j] = 0;
      }

      float *scale_data = scale_temp_ + tid * this->num_output_ * pooled_height_ * pooled_width_;

#pragma omp for
      for (int n = 0; n < this->num_; ++n) { // JSP: this->num_ is batch size
        float *conv_top_data = conv_top + n * this->top_dim_;

        // Convolution
        this->forward_cpu_gemm(
              bottom_data + n * this->bottom_dim_, weight, conv_top_data, n);
        if (this->bias_term_ &&
            this->layer_param_.convolution_param().conv_mode() != caffe::ConvolutionParameter_ConvMode_DIRECT_DCONV &&
            this->layer_param_.convolution_param().conv_mode() != caffe::ConvolutionParameter_ConvMode_DIRECT_SCONV) {
          // bias term is fused with direct convolution

          // JSP: common path of AlexNet
          if (0 == tid) bias_time -= omp_get_wtime();
          // conv_top_data += bias outer-prod bias_mult
          // (# of output)x(img size) += (# of output) outer-prod (img size)
          this->forward_cpu_bias(conv_top_data, bias);
          if (0 == tid) bias_time += omp_get_wtime();
        }

        // Pooling
        const int pool_top_count = pool_top_[0]->count();
        // We'll output the mask to top[1] if it's of size >1.
        const bool use_top_mask = pool_top_.size() > 1;
        int pool_top_offset = pool_top_[0]->offset(n);
        float *pool_top_data = pool_top + pool_top_offset;

        using std::min;
        using std::max;

        if (0 == tid) pool_time -= omp_get_wtime();

        if (this->layer_param_.convolution_param().conv_mode() != caffe::ConvolutionParameter_ConvMode_DIRECT_DCONV &&
            this->layer_param_.convolution_param().conv_mode() != caffe::ConvolutionParameter_ConvMode_DIRECT_SCONV) {
          // pooling is fused with direct convolution

          switch (this->layer_param_.pooling_param().pool()) {
          case PoolingParameter_PoolMethod_MAX:
            // The main loop
            int len = conv_top_[i]->offset(0, 1);
            for (int c = 0; c < this->num_output_; ++c) {
              // compute offset
              float *conv_top_data_cur = conv_top_data + len*c;
              float *pool_top_data_cur = pool_top_data + pool_top_[0]->offset(0, 1)*c;

              if (use_top_mask) {
                float *pool_top_mask_cur = pool_top_mask + pool_top_offset + pool_top_[0]->offset(0, 1)*c;

                for (int ph = 0; ph < pooled_height_; ++ph) {
                  for (int pw = 0; pw < pooled_width_; ++pw) {
                    int hstart = ph * stride_h_ - pad_h_;
                    int wstart = pw * stride_w_ - pad_w_;
                    int hend = min(hstart + kernel_h_, height_);
                    int wend = min(wstart + kernel_w_, width_);
                    hstart = max(hstart, 0);
                    wstart = max(wstart, 0);
                    float maximum = 0;
                    int mask = -1;
                    for (int h = hstart; h < hend; ++h) {
                      for (int w = wstart; w < wend; ++w) {
                        const int index = h * width_ + w;
                        if (conv_top_data_cur[index] > maximum) {
                          maximum = conv_top_data_cur[index];
                          mask = static_cast<float>(index);
                        }
                      }
                    }
                    const int pool_index = ph * pooled_width_ + pw;
                    pool_top_data_cur[pool_index] = maximum;
                    pool_top_data_cur[pool_index] = mask;
                  }
                }
              }
              else {
                // JSP: common path for AlexNet (stride=2, kernel=3)
                int *mask_cur = mask + pool_top_offset + pool_top_[0]->offset(0, 1)*c;

                if (stride_h_ == 2 && stride_w_ == 2 && kernel_h_ == 3 && kernel_w_ == 3 && pad_h_ == 0 && pad_w_ == 0) {
                  const int STRIDE = 2;
                  const int K = 3;

                  for (int ph = 0; ph < (height_ - K)/STRIDE; ++ph) {
                    int hstart = ph * STRIDE;
                    int hend = hstart + K;

                    for (int pw = 0; pw < (width_ - K)/STRIDE; ++pw) {
                      int wstart = pw * STRIDE;
                      float maximum = 0; // JSP: using 0 instead of -FLT_MAX does ReLU for us.
                      int mask = -1;

                      int index = hstart * width_ + wstart;
                      if (conv_top_data_cur[index] > maximum) {
                        maximum = conv_top_data_cur[index];
                        mask = index;
                      }
                      index = hstart * width_ + wstart + 1;
                      if (conv_top_data_cur[index] > maximum) {
                        maximum = conv_top_data_cur[index];
                        mask = index;
                      }
                      index = hstart * width_ + wstart + 2;
                      if (conv_top_data_cur[index] > maximum) {
                        maximum = conv_top_data_cur[index];
                        mask = index;
                      }

                      index = (hstart + 1) * width_ + wstart;
                      if (conv_top_data_cur[index] > maximum) {
                        maximum = conv_top_data_cur[index];
                        mask = index;
                      }
                      index = (hstart + 1) * width_ + wstart + 1;
                      if (conv_top_data_cur[index] > maximum) {
                        maximum = conv_top_data_cur[index];
                        mask = index;
                      }
                      index = (hstart + 1) * width_ + wstart + 2;
                      if (conv_top_data_cur[index] > maximum) {
                        maximum = conv_top_data_cur[index];
                        mask = index;
                      }

                      index = (hstart + 2) * width_ + wstart;
                      if (conv_top_data_cur[index] > maximum) {
                        maximum = conv_top_data_cur[index];
                        mask = index;
                      }
                      index = (hstart + 2) * width_ + wstart + 1;
                      if (conv_top_data_cur[index] > maximum) {
                        maximum = conv_top_data_cur[index];
                        mask = index;
                      }
                      index = (hstart + 2) * width_ + wstart + 2;
                      if (conv_top_data_cur[index] > maximum) {
                        maximum = conv_top_data_cur[index];
                        mask = index;
                      }

                      const int pool_index = ph * pooled_width_ + pw;
                      pool_top_data_cur[pool_index] = maximum;
                      mask_cur[pool_index] = mask;
                    }

                    for (int pw = (width_ - K)/STRIDE; pw < pooled_width_; ++pw) {
                      int wstart = pw * STRIDE;
                      int wend = min(wstart + K, width_);
                      float maximum = 0; // JSP: using 0 instead of -FLT_MAX does ReLU for us.
                      int mask = -1;
                      for (int h = hstart; h < hend; ++h) {
                        for (int w = wstart; w < wend; ++w) {
                          const int index = h * width_ + w;
                          if (conv_top_data_cur[index] > maximum) {
                            maximum = conv_top_data_cur[index];
                            mask = index;
                          }
                        }
                      }
                      const int pool_index = ph * pooled_width_ + pw;
                      pool_top_data_cur[pool_index] = maximum;
                      mask_cur[pool_index] = mask;
                    }
                  }

                  for (int ph = (height_ - K)/STRIDE; ph < pooled_height_; ++ph) {
                    int hstart = ph * STRIDE;
                    int hend = min(hstart + K, height_);

                    for (int pw = 0; pw < pooled_width_; ++pw) {
                      int wstart = pw * STRIDE;
                      int wend = min(wstart + K, width_);
                      float maximum = 0; // JSP: using 0 instead of -FLT_MAX does ReLU for us.
                      int mask = -1;
                      for (int h = hstart; h < hend; ++h) {
                        for (int w = wstart; w < wend; ++w) {
                          const int index = h * width_ + w;
                          if (conv_top_data_cur[index] > maximum) {
                            maximum = conv_top_data_cur[index];
                            mask = index;
                          }
                        }
                      }
                      const int pool_index = ph * pooled_width_ + pw;
                      pool_top_data_cur[pool_index] = maximum;
                      mask_cur[pool_index] = mask;
                    }
                  }
                }
                else {
                  LOG(WARNING) << "Inefficient code path";
                  for (int ph = 0; ph < pooled_height_; ++ph) {
                    int hstart = max(ph * stride_h_ - pad_h_, 0);
                    int hend = min(hstart + kernel_h_, height_);

                    for (int pw = 0; pw < pooled_width_; ++pw) {
                      int wstart = max(pw * stride_w_ - pad_w_, 0);
                      int wend = min(wstart + kernel_w_, width_);
                      float maximum = 0; // JSP: using 0 instead of -FLT_MAX does ReLU for us.
                      int mask = -1;
                      for (int h = hstart; h < hend; ++h) {
                        for (int w = wstart; w < wend; ++w) {
                          const int index = h * width_ + w;
                          if (conv_top_data_cur[index] > maximum) {
                            maximum = conv_top_data_cur[index];
                            mask = index;
                          }
                        }
                      }
                      const int pool_index = ph * pooled_width_ + pw;
                      pool_top_data_cur[pool_index] = maximum;
                      mask_cur[pool_index] = mask;
                    }
                  }
                }
              } // !use_pool_top_mask
            } // for each channel
            break;
          case PoolingParameter_PoolMethod_AVE:
            NOT_IMPLEMENTED;
            break;
          case PoolingParameter_PoolMethod_STOCHASTIC:
            NOT_IMPLEMENTED;
            break;
          default:
            LOG(FATAL) << "Unknown pooling method.";
          }
        }

        if (0 == tid) {
          pool_time += omp_get_wtime();
          lrn_time -= omp_get_wtime();
        }

        int offset = scale_.offset(n, 0);

        // pooled_height is 13 or 27

        int temp_mask1[8] = { -1, -1, -1, -1, -1, 0, 0, 0, };
        int temp_mask2[8] = { -1, -1, -1,  0,  0, 0, 0, 0, };
        __m256i temp_mask_v;
        if (pooled_height_ == 13) {
          temp_mask_v = _mm256_load_si256((__m256i *)temp_mask1);
        }
        else {
          assert(pooled_height_ == 27);
          temp_mask_v = _mm256_load_si256((__m256i *)temp_mask2);
        }

        for (int i = 0; i < pooled_height_; ++i) {
          // compute the padded square
          for (int c = pre_pad_; c < this->num_output_ + pre_pad_; ++c) {
            for (int j = 0; j < pooled_width_; ++j) {
              float d = pool_top_data[((c - pre_pad_) * pooled_height_ + i) * pooled_width_ + j];
              padded_square_data[c * pooled_width_ + j] = d * d;
            }
          }

          // Create the first channel scale
          for (int j = 0; j < pooled_width_; ++j) {
            scale_data[i * pooled_width_ + j] = k_ + alpha_over_size*padded_square_data[j];
          }
          for (int c = 1; c < size_; ++c) {
            for (int j = 0; j < pooled_width_; ++j) {
              scale_data[i * pooled_width_ + j] += alpha_over_size*padded_square_data[c * pooled_width_ + j];
            }
          }

          for (int c = 1; c < this->num_output_; ++c) {
            int offset = ((c - 1)*pooled_height_ + i)*pooled_width_;

            for (int j = 0; j < pooled_width_; ++j) {
              scale_data[(c * pooled_height_ + i) * pooled_width_ + j] =
                  scale_data[offset + j] +
                alpha_over_size*(
                    padded_square_data[(c + size_ - 1) * pooled_width_ + j] -
                    padded_square_data[(c - 1) * pooled_width_ + j]);
            }

//            if (pooled_height_ == 13) {
//              _mm256_storeu_ps(
//                  scale_data + offset,
//                  _mm256_pow_ps(_mm256_loadu_ps(scale_data + offset), _mm256_set1_ps(-beta_)));
//              _mm256_maskstore_ps(
//                  scale_data + offset + 8, temp_mask_v,
//                  _mm256_pow_ps(_mm256_loadu_ps(scale_data + offset + 8), _mm256_set1_ps(-beta_)));
//            }
//            else {
//              assert(pooled_height_ == 27);
//              _mm256_storeu_ps(
//                  scale_data + offset,
//                  _mm256_pow_ps(_mm256_loadu_ps(scale_data + offset), _mm256_set1_ps(-beta_)));
//              _mm256_storeu_ps(
//                  scale_data + offset + 8,
//                  _mm256_pow_ps(_mm256_loadu_ps(scale_data + offset + 8), _mm256_set1_ps(-beta_)));
//              _mm256_storeu_ps(
//                  scale_data + offset + 16,
//                  _mm256_pow_ps(_mm256_loadu_ps(scale_data + offset + 16), _mm256_set1_ps(-beta_)));
//              _mm256_maskstore_ps(
//                  scale_data + offset + 24, temp_mask_v,
//                  _mm256_pow_ps(_mm256_loadu_ps(scale_data + offset + 24), _mm256_set1_ps(-beta_)));
//            }
          }
//          int offset = ((this->num_output_ - 1)*pooled_height_ + i)*pooled_width_;
//          if (pooled_height_ == 13) {
//            _mm256_storeu_ps(
//                scale_data + offset,
//                _mm256_pow_ps(_mm256_loadu_ps(scale_data + offset), _mm256_set1_ps(-beta_)));
//            _mm256_maskstore_ps(
//                scale_data + offset + 8, temp_mask_v,
//                _mm256_pow_ps(_mm256_loadu_ps(scale_data + offset + 8), _mm256_set1_ps(-beta_)));
//          }
//          else {
//            assert(pooled_height_ == 27);
//            _mm256_storeu_ps(
//                scale_data + offset,
//                _mm256_pow_ps(_mm256_loadu_ps(scale_data + offset), _mm256_set1_ps(-beta_)));
//            _mm256_storeu_ps(
//                scale_data + offset + 8,
//                _mm256_pow_ps(_mm256_loadu_ps(scale_data + offset + 8), _mm256_set1_ps(-beta_)));
//            _mm256_storeu_ps(
//                scale_data + offset + 16,
//                _mm256_pow_ps(_mm256_loadu_ps(scale_data + offset + 16), _mm256_set1_ps(-beta_)));
//            _mm256_maskstore_ps(
//                scale_data + offset + 24, temp_mask_v,
//                _mm256_pow_ps(_mm256_loadu_ps(scale_data + offset + 24), _mm256_set1_ps(-beta_)));
//          }
        }

        for (int i = 0; i < this->num_output_ * pooled_height_ * pooled_width_; i += 8) {
          __m256 v = _mm256_pow_ps(_mm256_load_ps(scale_data + i), _mm256_set1_ps(-beta_));
          v = _mm256_mul_ps(v, _mm256_load_ps(pool_top + offset + i));
          _mm256_store_ps(top_data + offset + i, v);
        }

        if (0 == tid) lrn_time += omp_get_wtime();
      } // for (int n = 0; n < this->num_; ++n)
    } // omp parallel
  } // for (int i = 0; i < bottom.size(); ++i)

  if (this->layer_param_.convolution_param().conv_mode() == caffe::ConvolutionParameter_ConvMode_DIRECT_SCONV) {
    int pad_h = this->pad_.cpu_data()[0];
    int pad_w = this->pad_.cpu_data()[1];

    if (pad_h != 0 || pad_w != 0) {
      LOG(INFO) << "padding " << padding_time*1e3 << " ms";
    }
  }
  else {
    LOG(INFO) << "im2col " << im2col_time*1e3 << " ms";
  }
  if (this->layer_param_.convolution_param().conv_mode() == caffe::ConvolutionParameter_ConvMode_DIRECT_DCONV ||
      this->layer_param_.convolution_param().conv_mode() == caffe::ConvolutionParameter_ConvMode_DIRECT_SCONV) {
    LOG(INFO) << "transpose " << transpose_cycle << " pool " << ::pool_cycle;
  }

  LOG(INFO) << "bias " << bias_time*1e3 << " ms, pool " << pool_time*1e3 << " ms, lrn " << lrn_time*1e3 << " ms";

  int height = this->conv_input_shape_.cpu_data()[1];
  int width = this->conv_input_shape_.cpu_data()[2];
  int kernel_h = this->kernel_shape_.cpu_data()[0];
  int kernel_w = this->kernel_shape_.cpu_data()[1];
  int pad_h = this->pad_.cpu_data()[0];
  int pad_w = this->pad_.cpu_data()[1];
  int stride_h = this->stride_.cpu_data()[0];
  int stride_w = this->stride_.cpu_data()[1];
  int dilation_h = this->dilation_.cpu_data()[0];
  int dilation_w = this->dilation_.cpu_data()[1];

  const int output_h = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  double flops = (double)this->num_*this->conv_out_channels_*this->conv_in_channels_/this->group_*output_h*output_w*kernel_h*kernel_w*2;

  unsigned long long max_conv_cycle = 0, sum_conv_cycle = 0;
  for (int i = 0; i < omp_get_max_threads(); ++i) {
    max_conv_cycle = std::max(max_conv_cycle, conv_cycles_of_this_batch[i*16]);
    sum_conv_cycle += conv_cycles_of_this_batch[i*16];
  }
  std::string name(this->layer_param_.name());
  LOG(INFO) <<
      name <<
      " K-cycles-per-file max " << max_conv_cycle/1000./this->num_ <<
      " avg " << sum_conv_cycle/1000./omp_get_max_threads()/this->num_ <<
      " mFlops-per-file " << flops/this->num_/1e6 <<
      " GF/s " << flops/(max_conv_cycle/get_cpu_freq())/1e9;

  if (total_conv_cycles.find(name) == total_conv_cycles.end()) {
    total_conv_cycles[name] = 0;
    total_conv_flops[name] = 0;
  }
  total_conv_cycles[name] += max_conv_cycle;
  total_conv_flops[name] += flops;
  total_files += this->num_;
}

template<>
void ConvolutionReLUPoolLRNLayer<double>::Forward_cpu(const vector<Blob<double>*>& bottom,
      const vector<Blob<double>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void ConvolutionReLUPoolLRNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  assert(false); // TODO
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionReLUPoolLRNLayer);
#endif

INSTANTIATE_CLASS(ConvolutionReLUPoolLRNLayer);
REGISTER_LAYER_CLASS(ConvolutionReLUPoolLRN);

}  // namespace caffe
