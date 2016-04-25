#include <vector>
#include <omp.h>

#include "caffe/layers/conv_relu_pool_layer.hpp"

extern unsigned long long conv_cycles_of_this_batch[1024*16];
extern std::map<std::string, unsigned long long> total_conv_cycles;
extern std::map<std::string, double> total_conv_flops;
extern int total_files;

double get_cpu_freq();

namespace caffe {

template <typename Dtype>
void ConvolutionReLUPoolLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
   BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, middle_);

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
}

template <typename Dtype>
void ConvolutionReLUPoolLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (middle_.empty()) {
     middle_.resize(bottom.size());
     for (int i = 0; i < middle_.size(); ++i) {
        middle_[i] = new Blob<Dtype>();
     }
  }
  BaseConvolutionLayer<Dtype>::Reshape(bottom, middle_);

  CHECK_EQ(4, middle_[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
//  channels_ = middle_[0]->channels();
  height_ = middle_[0]->height();
  width_ = middle_[0]->width();
  if (global_pooling_) {
    kernel_h_ = middle_[0]->height();
    kernel_w_ = middle_[0]->width();
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
  top[0]->Reshape(middle_[0]->num(), this->num_output_, pooled_height_,
      pooled_width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(middle_[0]->num(), this->num_output_, pooled_height_,
        pooled_width_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(middle_[0]->num(), this->num_output_, pooled_height_,
      pooled_width_);
  }
}

template <typename Dtype>
void ConvolutionReLUPoolLayer<Dtype>::compute_output_shape() {
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

template <typename Dtype>
void ConvolutionReLUPoolLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype *bias = NULL;
  if (this->bias_term_) {
    bias = this->blobs_[1]->cpu_data();
  }

  for (int i = 0; i < omp_get_max_threads(); ++i) {
    conv_cycles_of_this_batch[i*16] = 0;
  }

  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  assert(negative_slope == 0);

  Dtype* top_data = top[0]->mutable_cpu_data();
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  const bool use_top_mask = top.size() > 1;
  if (use_top_mask) {
    top_mask = top[1]->mutable_cpu_data();
  }
  else {
    mask = max_idx_.mutable_cpu_data();
  }

  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* middle_data = middle_[i]->mutable_cpu_data();
#pragma omp parallel for
    for (int n = 0; n < this->num_; ++n) { // JSP: this->num_ is batch size
      Dtype *middle_current = middle_data + n * this->top_dim_;

      // Convolution
      this->forward_cpu_gemm(
            bottom_data + n * this->bottom_dim_, weight, middle_current, n);
      if (this->bias_term_) {
        // JSP: common path of AlexNet
        this->forward_cpu_bias(middle_current, bias);
      }

      // Pooling
      const int top_count = top[0]->count();
      // We'll output the mask to top[1] if it's of size >1.

      using std::min;
      using std::max;

      switch (this->layer_param_.pooling_param().pool()) {
      case PoolingParameter_PoolMethod_MAX:
        // The main loop
        int len = middle_[i]->offset(0, 1);
        for (int c = 0; c < this->num_output_; ++c) {
          // compute offset
          Dtype *middle_data_cur = middle_data + len*(this->num_output_*n + c);
          Dtype *top_data_cur = top_data + top[0]->offset(0, 1)*(this->num_output_*n + c);

          if (use_top_mask) {
            Dtype *top_mask_cur = top_mask + top[0]->offset(0, 1)*(this->num_output_*n + c);

            for (int ph = 0; ph < pooled_height_; ++ph) {
              for (int pw = 0; pw < pooled_width_; ++pw) {
                int hstart = ph * stride_h_ - pad_h_;
                int wstart = pw * stride_w_ - pad_w_;
                int hend = min(hstart + kernel_h_, height_);
                int wend = min(wstart + kernel_w_, width_);
                hstart = max(hstart, 0);
                wstart = max(wstart, 0);
                Dtype maximum = 0;
                int mask = -1;
                for (int h = hstart; h < hend; ++h) {
                  for (int w = wstart; w < wend; ++w) {
                    const int index = h * width_ + w;
                    if (middle_data_cur[index] > maximum) {
                      maximum = middle_data_cur[index];
                      mask = static_cast<Dtype>(index);
                    }
                  }
                }
                const int pool_index = ph * pooled_width_ + pw;
                top_data_cur[pool_index] = maximum;
                top_data_cur[pool_index] = mask;
              }
            }
          }
          else {
            // JSP: common path for AlexNet (stride=2, kernel=3)
            int *mask_cur = mask + top[0]->offset(0, 1)*(this->num_output_*n + c);

            for (int ph = 0; ph < pooled_height_; ++ph) {
              int hstart = max(ph * stride_h_ - pad_h_, 0);
              int hend = min(hstart + kernel_h_, height_);

              for (int pw = 0; pw < pooled_width_; ++pw) {
                int wstart = max(pw * stride_w_ - pad_w_, 0);
                int wend = min(wstart + kernel_w_, width_);
                Dtype maximum = 0; // JSP: using 0 instead of -FLT_MAX does ReLU for us.
                int mask = -1;
                for (int h = hstart; h < hend; ++h) {
                  for (int w = wstart; w < wend; ++w) {
                    const int index = h * width_ + w;
                    if (middle_data_cur[index] > maximum) {
                      maximum = middle_data_cur[index];
                      mask = index;
                    }
                  }
                }
                const int pool_index = ph * pooled_width_ + pw;
                top_data_cur[pool_index] = maximum;
                mask_cur[pool_index] = mask;
              }
            }
          } // !use_top_mask
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
    } // for (int n = 0; n < this->num_; ++n)
  } // for (int i = 0; i < bottom.size(); ++i)

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

template <typename Dtype>
void ConvolutionReLUPoolLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  assert(false); // TODO
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionReLUPoolLayer);
#endif

INSTANTIATE_CLASS(ConvolutionReLUPoolLayer);
REGISTER_LAYER_CLASS(ConvolutionReLUPool);

}  // namespace caffe
