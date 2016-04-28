#include <vector>
#include <omp.h>

#include "caffe/layers/conv_relu_layer.hpp"
#include "caffe/util/math_functions_intel.hpp"
#include "caffe/util/conv.hpp"

extern unsigned long long conv_cycles_of_this_batch[1024*16];
extern std::map<std::string, unsigned long long> total_conv_cycles;
extern std::map<std::string, double> total_conv_flops;
extern int total_files;

double get_cpu_freq();

namespace caffe {

extern double padding_time;

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
  for (int i = 0; i < omp_get_max_threads(); ++i) {
    conv_cycles_of_this_batch[i*16] = 0;
  }

  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype *bias = NULL;
  if (this->bias_term_) {
    bias = this->blobs_[1]->cpu_data();
  }
  double t = omp_get_wtime();
  double t2 = 0, t3 = 0;
  padding_time = 0;
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
#pragma omp parallel
    {
      int nthreads = omp_get_num_threads();
      int tid = omp_get_thread_num();

      int nthread_groups = nthreads;
#ifdef __AVX512F__
      nthread_groups = NTILES;
#else
//      nthread_groups /= 2; // 1 group per core in Xeon
#endif
      assert(nthreads%nthread_groups == 0);
      int nthreads_per_group = nthreads/nthread_groups;
      int gid = tid/nthreads_per_group;

      int n_per_group = (this->num_ + nthread_groups - 1)/nthread_groups;
      int n_begin = std::min(n_per_group*gid, this->num_);
      int n_end = std::min(n_begin + n_per_group, this->num_);

      for (int n = n_begin; n < n_end; ++n) { // JSP: this->num_ is batch size
        Dtype *top_current = top_data + n * this->top_dim_;

        if (0 == tid) t2 -= omp_get_wtime();
        this->forward_cpu_gemm(
              bottom_data + n * this->bottom_dim_, weight, top_current, n);
        if (0 == tid) t2 += omp_get_wtime();
        if (this->bias_term_ &&
            this->layer_param_.convolution_param().conv_mode() != caffe::ConvolutionParameter_ConvMode_DIRECT_SCONV) {
          // JSP: common path of AlexNet
          this->forward_cpu_bias(top_current, bias);
        }

        if (0 == tid) t3 -= omp_get_wtime();

        int j_per_thread = (this->top_dim_ + nthreads_per_group - 1)/nthreads_per_group;
        int tid_in_group = tid%nthreads_per_group;
        int jbegin = std::min(j_per_thread*tid_in_group, this->top_dim_);
        int jend = std::min(jbegin + j_per_thread, this->top_dim_);

        if (nthread_groups != nthreads) barriers[gid]->wait(tid_in_group);

        if (negative_slope == 0) {
          for (int j = jbegin; j < jend; ++j) {
            top_current[j] = std::max(top_current[j], Dtype(0));
          }
        }
        else {
          for (int j = jbegin; j < jend; ++j) {
            top_current[j] =
                 std::max(top_current[j], Dtype(0)) +
                 negative_slope * std::min(top_current[j], Dtype(0));
          }
        }

        if (0 == tid) t3 += omp_get_wtime();
      }
    }
  }

  LOG(INFO) << this->layer_param_.name() << " wall clock-time " << omp_get_wtime() - t << " padding-time " << padding_time << " relu-time " << t3;

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
