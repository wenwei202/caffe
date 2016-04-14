#include <vector>

#include "caffe/layers/conv_relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionReLULayer<Dtype>::compute_output_shape() {
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
void ConvolutionReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
#pragma omp parallel for
    for (int n = 0; n < this->num_; ++n) { // JSP: this->num_ is batch size
      Dtype *top_current = top_data + n * this->top_dim_;

      this->forward_cpu_gemm(
            bottom_data + n * this->bottom_dim_, weight, top_current, n);
      if (this->bias_term_) {
        // JSP: common path of AlexNet
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_current, bias);
      }

      if (negative_slope == 0) {
         for (int j = 0; j < this->top_dim_; ++j) {
            top_current[j] = std::max(top_current[j], Dtype(0));
         }
      }
      else {
         for (int j = 0; j < this->top_dim_; ++j) {
            top_current[j] =
                  std::max(top_current[j], Dtype(0)) +
                  negative_slope * std::min(top_current[j], Dtype(0));
         }
      }
    }
  }
}

template <typename Dtype>
void ConvolutionReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  assert(false); // TODO
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionReLULayer);
#endif

INSTANTIATE_CLASS(ConvolutionReLULayer);
REGISTER_LAYER_CLASS(ConvolutionReLU);

}  // namespace caffe
