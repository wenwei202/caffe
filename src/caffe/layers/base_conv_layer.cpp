#include <algorithm>
#include <vector>
#include <omp.h>
#ifdef __INTEL_COMPILER
#include <immintrin.h>
#endif

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/math_functions_intel.hpp"
#include "caffe/layers/conv_relu_pool_lrn_layer.hpp"

#include "conv.hpp"

namespace caffe {

template <typename Dtype>
BaseConvolutionLayer<Dtype>::~BaseConvolutionLayer()
{
  free(weight_interleaved_);
  free(input_padded_);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::WeightAlign(){
	CHECK_EQ(this->blobs_[0]->num_axes(),4);//caffe now supports any dimension
	//is_sparse_format_weights_ = false;
	const LayerParameter& layerparam = this->layer_param();
	LOG(INFO)<<"layer\t"<<layerparam.name()<<"\t"<<"has sparsity of "<< this->blobs_[0]->GetSparsity();
	this->blobs_[0]->WriteToNistMMIO(layerparam.name()+".weight");

	ConvolutionParameter conv_param = this->layer_param_.convolution_param();
	const int M = this->blobs_[0]->shape(0)/group_;
	const int N = this->blobs_[0]->count(1,4);
	const int weight_offset = this->blobs_[0]->count()/group_;
	const int row_offset = this->blobs_[0]->shape(0)/group_ + 1;
	int masked_col_num = 0;
	int left_cols = 0;
	Dtype group_sparsity = 0;
	switch(conv_param.conv_mode()){
		case caffe::ConvolutionParameter_ConvMode_LOWERED_CSRMM:
			LOG(INFO)<<"ConvolutionParameter_ConvMode_LOWERED_CSRMM";
			for (int g = 0; g < group_; ++g) {
				caffe_cpu_sparse_dense2csr(M, N,
						this->blobs_[0]->mutable_cpu_data() + weight_offset * g,
						nz_weight_values_.mutable_cpu_data()+ weight_offset * g,
						nz_weight_indices_.mutable_cpu_data()+ weight_offset * g,
						nz_weight_index_pointers_.mutable_cpu_data() + row_offset * g);
			}
			break;
		case caffe::ConvolutionParameter_ConvMode_LOWERED_CCNMM:
			LOG(INFO)<<"ConvolutionParameter_ConvMode_LOWERED_CCNMM";
			for (int g = 0; g < group_; ++g) {
				caffe_cpu_if_all_zero(M,
						N,
						this->blobs_[0]->cpu_data() + weight_offset * g,
						col_buf_mask_.mutable_cpu_data() + N * g);
			}
			masked_col_num = 0;
			for(int idx=0; idx<col_buf_mask_.count();++idx){
				if(col_buf_mask_.cpu_data()[idx]){
					masked_col_num++;
				}
			}
			group_sparsity = (Dtype)masked_col_num/(Dtype)col_buf_mask_.count();
			LOG(INFO) << Layer<Dtype>::layer_param().name() << " column sparsity: " << group_sparsity;
			is_concatenating_weights_features_ = true;

			// compress weight matrix
			left_cols = 0;
			for (int g = 0; g < group_; ++g) {
				caffe_cpu_del_zero_cols(conv_out_channels_ /group_,
					  kernel_dim_ ,
					  this->blobs_[0]->cpu_data() + weight_offset_ * g,
					  squeezed_weight_buffer_.mutable_cpu_data() + weight_offset_ * g,
					  &left_cols,
					  col_buf_mask_.cpu_data() + kernel_dim_ * g );
				left_columns_.push_back(left_cols);
			}
			break;
		case caffe::ConvolutionParameter_ConvMode_DIRECT_SCONV:
			{
				LOG(INFO)<<"ConvolutionParameter_ConvMode_DIRECT_SCONV";

        int height = conv_input_shape_.cpu_data()[1];
        int width = conv_input_shape_.cpu_data()[2];
        int pad_h = pad_.cpu_data()[0];
        int pad_w = pad_.cpu_data()[1];

				for (int g = 0; g < group_; ++g) {
					// first create a CSR matrix as for LOWERED_CSRMM
					caffe_cpu_sparse_dense2csr(M, N,
							this->blobs_[0]->mutable_cpu_data() + weight_offset * g,
							nz_weight_values_.mutable_cpu_data()+ weight_offset * g,
							nz_weight_indices_.mutable_cpu_data()+ weight_offset * g,
							nz_weight_index_pointers_.mutable_cpu_data() + row_offset * g);

					int kernel_h = kernel_shape_.cpu_data()[0];
					int kernel_w = kernel_shape_.cpu_data()[1];

					// declare variables for sparsity statistics
					vector<vector<int> > nnz_per_channel_pair(M);
					for(int i = 0; i < M; ++i) {
						nnz_per_channel_pair[i] = vector<int>(conv_in_channels_, 0);
					}
					int num_of_non_zero_kernels = 0;

					// transform the indices for direct convolution
					const int *rowptr = nz_weight_index_pointers_.cpu_data() + row_offset * g;
					int *colidx = nz_weight_indices_.mutable_cpu_data() + weight_offset * g;
					for (int out_channel = 0; out_channel < M; ++out_channel) {
						for (int j = rowptr[out_channel]; j < rowptr[out_channel + 1]; ++j) {
							int col = colidx[j];

							int kernel_col = col%kernel_w;
							int kernel_row = (col/kernel_w)%kernel_h;
							int in_channel = col/(kernel_w*kernel_h);
							assert(in_channel < conv_in_channels_);

							colidx[j] = (in_channel*(height + pad_h) + kernel_row)*(width + pad_w) + kernel_col;

							nnz_per_channel_pair[out_channel][in_channel]++;
						}

						for (int in_channel = 0; in_channel < conv_in_channels_; ++in_channel) {
							if (nnz_per_channel_pair[out_channel][in_channel] != 0) {
								++num_of_non_zero_kernels;
							}
						}
					}

					LOG(INFO) << "k-mode sparsity " << (double)num_of_non_zero_kernels/(M*conv_in_channels_);
				}

				int input_padded_len = conv_in_channels_ * (height + pad_h) * (width + pad_w) + pad_h * (width + 2 * pad_w);
				posix_memalign((void **)&input_padded_, 4096, sizeof(float)*omp_get_max_threads()*input_padded_len);

				break;
			}
		case caffe::ConvolutionParameter_ConvMode_DIRECT_DCONV:
		  LOG(INFO)<<"ConvolutionParameter_ConvMode_DIRECT_DCONV";
		  {
        int kernel_h = kernel_shape_.cpu_data()[0];
        int kernel_w = kernel_shape_.cpu_data()[1];
//        int kernel_size_aligned = (kernel_h * kernel_w + 15)/16*16;
//        posix_memalign(
//            (void **)&weight_aligned_,
//            1024,
//            sizeof(Dtype) * M * conv_in_channels_ * kernel_size_aligned);
//        for (int g = 0; g < group_; ++g) {
//          for (int out_channel = 0; out_channel < M; ++out_channel) {
//            for (int in_channel = 0; in_channel < conv_in_channels_/group_; ++in_channel) {
//              memcpy(
//                  weight_aligned_ + ((g*M + out_channel)*(conv_in_channels_/group_) + in_channel)*kernel_size_aligned,
//                  this->blobs_[0]->cpu_data() + weight_offset*g + (out_channel*(conv_in_channels_/group_) + in_channel)*kernel_h*kernel_w,
//                  sizeof(Dtype)*kernel_h*kernel_w);
//            }
//          }
//        }
//
//        int kernel_w_aligned = (kernel_w + 7)/8*8;
//        int kernel_size_aligned2 = (kernel_h*kernel_w_aligned + 15)/16*16;
//        posix_memalign(
//            (void **)&weight_aligned2_,
//            1024,
//            sizeof(Dtype) * M * conv_in_channels_ * kernel_size_aligned2);
//        for (int g = 0; g < group_; ++g) {
//          for (int out_channel = 0; out_channel < M; ++out_channel) {
//            for (int in_channel = 0; in_channel < conv_in_channels_/group_; ++in_channel) {
//              for (int h = 0; h < kernel_h; ++h) {
//                memcpy(
//                    weight_aligned2_ + ((g*M + out_channel)*(conv_in_channels_/group_) + in_channel)*kernel_size_aligned2 + h*kernel_w_aligned,
//                    this->blobs_[0]->cpu_data() + weight_offset*g + ((out_channel*(conv_in_channels_/group_) + in_channel)*kernel_h + h)*kernel_w,
//                    sizeof(Dtype)*kernel_w);
//                memset(
//                    weight_aligned2_ + ((g*M + out_channel)*(conv_in_channels_/group_) + in_channel)*kernel_size_aligned2 + h*kernel_w_aligned + kernel_w,
//                    0,
//                    sizeof(Dtype)*(kernel_w_aligned - kernel_w));
//              }
//            }
//          }
//        }

        assert(M%8 == 0);
        posix_memalign(
            (void **)&weight_interleaved_,
            4096,
            sizeof(Dtype) * M * conv_in_channels_ * kernel_h * kernel_w);
        for (int g = 0; g < group_; ++g) {
          for (int out_channel_begin = 0; out_channel_begin < M; out_channel_begin += 8) {
            for (int in_channel = 0; in_channel < conv_in_channels_/group_; ++in_channel) {
              for (int k = 0; k < kernel_h*kernel_w; ++k) {
                for (int out_channel = out_channel_begin; out_channel < out_channel_begin + 8; ++out_channel) {
                  weight_interleaved_[(((g*M + out_channel_begin/8)*(conv_in_channels_/group_) + in_channel)*kernel_h*kernel_w + k)*8 + out_channel - out_channel_begin] =
                      this->blobs_[0]->cpu_data()[weight_offset*g + (out_channel*(conv_in_channels_/group_) + in_channel)*kernel_h*kernel_w + k];
                }
              }
            }
          }
        }
		  }
		  break;
		default:
			LOG(INFO)<<"ConvolutionParameter ConvMode: DEFAULT";
			break;
	}

	//disconnect connections
	if( layerparam.connectivity_mode() == caffe::LayerParameter_ConnectivityMode_DISCONNECTED_ELTWISE ){
		LOG(INFO)<<"all zero weights of "<<layerparam.name()<<" are frozen";
		this->blobs_[0]->Disconnect(Blob<Dtype>::ELTWISE);
	}else if(layerparam.connectivity_mode() == caffe::LayerParameter_ConnectivityMode_DISCONNECTED_GRPWISE){
		LOG(INFO)<<"weights lying in all-zero groups of "<<layerparam.name()<<" are frozen";
		this->blobs_[0]->Disconnect(Blob<Dtype>::GRPWISE, group_);
	}

}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  force_nd_im2col_ = conv_param.force_nd_im2col();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_ / group_;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  kernel_dim_ = this->blobs_[0]->count(1);
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  is_concatenating_weights_features_ = false;
  dense_feature_map_mask_.Reshape(1,1,1,channels_);
  squeezed_weight_buffer_.Reshape(this->blobs_[0]->shape(0),this->blobs_[0]->shape(1),this->blobs_[0]->shape(2),this->blobs_[0]->shape(3));
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (reverse_dimensions()) {
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_ * group_ * num_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (reverse_dimensions()) {
      col_buffer_shape_.push_back(input_shape(i + 1));
    } else {
      col_buffer_shape_.push_back(output_shape_[i]);
    }
  }
  col_buffer_.Reshape(col_buffer_shape_);

  if(!reverse_dimensions()){
	  col_buf_mask_.Reshape(1,1,1,kernel_dim_*group_);
#ifdef	GPU_USE_CUSPARSE
	  nonzero_elements_buffer_.Reshape(1, 1, 1, col_buffer_.count());//WARNING: real sparse matrix needs many less memory
	  nonzero_indices_buffer_.Reshape(1,1,1,nonzero_elements_buffer_.count());
	  index_pointers_buffer_.Reshape(1,1,1,col_buffer_.shape(1)+1);
	  nonzero_per_rowcol_buffer_.Reshape(1,1,1,col_buffer_.shape(1));
#endif
	  nz_weight_values_.Reshape(1, 1, 1, this->blobs_[0]->count());//nonzero elements
	  nz_weight_indices_.Reshape(1,1,1,nz_weight_values_.count());//index of nonzero
	  nz_weight_index_pointers_.Reshape(1,1,1,this->blobs_[0]->shape(0)+group_);//pointer(index) of indices
  }

  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
}

template <>
void BaseConvolutionLayer<double>::forward_cpu_gemm(const double* input,
    const double* weights, double* output, int batch_idx, bool skip_im2col) {
  NOT_IMPLEMENTED;
}

extern double padding_time, im2col_time;
unsigned long long conv_time, transpose_time, pool_time;

template<>
void BaseConvolutionLayer<float>::forward_cpu_gemm(const float* input,
    const float* weights, float* output, int batch_idx, bool skip_im2col) {
  const float* col_buff = input;

  Timer timer;
  timer.Start();

  float *input_padded;
  int input_padded_len;
  if (this->layer_param_.convolution_param().conv_mode() == caffe::ConvolutionParameter_ConvMode_DIRECT_SCONV) {
    // JSP: pad boundaries with zeros to avoid checking boundary conditions

	  int height = conv_input_shape_.cpu_data()[1];
	  int width = conv_input_shape_.cpu_data()[2];
	  int pad_h = pad_.cpu_data()[0];
	  int pad_w = pad_.cpu_data()[1];

	  input_padded_len = conv_in_channels_ * (height + pad_h) * (width + pad_w) + pad_h * (width + 2 * pad_w);
	  if (pad_h == 0 && pad_w == 0) {
	    input_padded = (float *)input;
	  }
	  else {
	    if (omp_get_thread_num() == 0) padding_time -= omp_get_wtime();

	    input_padded = input_padded_ + input_padded_len*omp_get_thread_num();
      for (int in_channel = 0; in_channel < conv_in_channels_; ++in_channel) {
        memset(
            input_padded + in_channel * (height + pad_h) * (width + pad_w),
            0, sizeof(float) * pad_h * (width + pad_w));
        for (int input_row = 0; input_row < height; ++input_row) {
          memset(
              input_padded + (in_channel * (height + pad_h) + input_row + pad_h) * (width + pad_w),
              0, sizeof(float) * pad_w);
          memcpy(
              input_padded + (in_channel * (height + pad_h) + input_row + pad_h) * (width + pad_w) + pad_w,
              input + (in_channel * height + input_row) * width,
              sizeof(float) * width);
        }
      }
      memset(
          input_padded + conv_in_channels_ * (height + pad_h) * (width + pad_w),
          0,
          sizeof(float) * pad_h * (width + 2 * pad_w));

      if (omp_get_thread_num() == 0) padding_time += omp_get_wtime();
	  }
  }
  else if (this->layer_param_.convolution_param().conv_mode() == caffe::ConvolutionParameter_ConvMode_DIRECT_DCONV) {
    // JSP: currently, don't need to do anything
  }
  else if (!is_1x1_ ||  is_concatenating_weights_features_) {
    if (omp_get_thread_num() == 0) im2col_time -= omp_get_wtime();

    int offset = 0;
    if (!skip_im2col || is_concatenating_weights_features_) {
      offset = col_offset_*group_*batch_idx;
      float *col_buff_mutable = col_buffer_.mutable_cpu_data() + offset;
      if(is_concatenating_weights_features_){
    	  conv_im2col_cpu(input, col_buff_mutable, col_buf_mask_.mutable_cpu_data()/*, dense_feature_map_mask_.mutable_cpu_data()*/);
      }else{
    	  conv_im2col_cpu(input, col_buff_mutable);
      }
    }
    col_buff = col_buffer_.cpu_data() + offset;

    if (omp_get_thread_num() == 0) im2col_time += omp_get_wtime();
  }

  int offset_sum = 0;
  Timer total_timer;
  total_timer.Start();
  for (int g = 0; g < group_; ++g) {
	  const int M = conv_out_channels_ /group_;
	  const int N = conv_out_spatial_dim_;
	  const int K = kernel_dim_;
	  const int row_offset = conv_out_channels_ /group_ + 1;
	  int left_cols = 0;
	  switch(this->layer_param_.convolution_param().conv_mode()){
	  case caffe::ConvolutionParameter_ConvMode_LOWERED_CSRMM :
		  timer.Start();
		  caffe_cpu_sparse_mmcsr(M,
				  N,
				  K,
				  (float)1.,
				  nz_weight_values_.cpu_data()+ weight_offset_ * g,
				  nz_weight_indices_.cpu_data()+ weight_offset_ * g,
				  nz_weight_index_pointers_.cpu_data() + row_offset * g,
				  nz_weight_index_pointers_.cpu_data() + row_offset * g + 1,
				  col_buff + col_offset_ * g,
				  (float)0.,output + output_offset_ * g);
		  break;
	  case caffe::ConvolutionParameter_ConvMode_LOWERED_CCNMM :
		  timer.Start();
		  left_cols = left_columns_[g];
		  caffe_cpu_cblas_gemm(conv_out_channels_ /
				  group_, conv_out_spatial_dim_, left_cols,
				  (float)1., squeezed_weight_buffer_.cpu_data() + weight_offset_ * g,
				  kernel_dim_ , col_buff + offset_sum,
				conv_out_spatial_dim_, (float)0., output + output_offset_ * g, conv_out_spatial_dim_);
		  offset_sum += left_cols * conv_out_spatial_dim_;
	  	  break;
	  case caffe::ConvolutionParameter_ConvMode_DIRECT_DCONV:
	  {
      int height = conv_input_shape_.cpu_data()[1];
      int width = conv_input_shape_.cpu_data()[2];
      int kernel_h = kernel_shape_.cpu_data()[0];
      int kernel_w = kernel_shape_.cpu_data()[1];
      int pad_h = pad_.cpu_data()[0];
      int pad_w = pad_.cpu_data()[1];
      int stride_h = stride_.cpu_data()[0];
      int stride_w = stride_.cpu_data()[1];
      int dilation_h = dilation_.cpu_data()[0];
      int dilation_w = dilation_.cpu_data()[1];

      if (height == 227 && width == 227 && pad_h == 0 && pad_w == 0 && stride_h == 4 && stride_w == 4 && kernel_w == 11 && kernel_h == 11 && dilation_h == 1 && dilation_w == 1 && conv_in_channels_/group_ == 3) {
        //      const int weight_offset = this->blobs_[0]->count()/group_;
        //      const float *weight = this->blobs_[0]->cpu_data() + weight_offset * g;
        //      int kernel_size_aligned = (kernel_h * kernel_w + 15)/16*16;
        //      const float *weight = weight_aligned_ + g*M*(conv_in_channels_/group_)*kernel_size_aligned;
        //      int kernel_w_aligned = (kernel_w + 7)/8*8;
        //      int kernel_size_aligned2 = (kernel_w_aligned*kernel_h + 15)/16*16;
        //      const float *weight = weight_aligned2_ + g*M*(conv_in_channels_/group_)*kernel_size_aligned2;

        const float *weight = weight_interleaved_;
        caffe_cpu_dconv<float>(
            input + conv_in_channels_/group_ * g * height * width,
            conv_in_channels_, height, width,
            pad_h, pad_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            weight_interleaved_,
            kernel_h, kernel_w,
            this->blobs_[1]->cpu_data(), bias_multiplier_.cpu_data(),
            ((ConvolutionReLUPoolLRNLayer<float> *)this)->pool_top_[0]->mutable_cpu_data() + ((ConvolutionReLUPoolLRNLayer<float> *)this)->pool_top_[0]->offset(batch_idx),
            ((ConvolutionReLUPoolLRNLayer<float> *)this)->max_idx_.mutable_cpu_data() + ((ConvolutionReLUPoolLRNLayer<float> *)this)->pool_top_[0]->offset(batch_idx),
            output,
            M + output_offset_ * g);
      }
      else {
        const int weight_offset = this->blobs_[0]->count()/group_;

        caffe_cpu_dconv<float>(
            input + conv_in_channels_/group_ * g * height * width,
            conv_in_channels_, height, width,
            pad_h, pad_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            weights + weight_offset * g,
            kernel_h, kernel_w,
            NULL, NULL, NULL, NULL,
            output + output_offset_ * g,
            M);
      }
      break;
	  }
	  case caffe::ConvolutionParameter_ConvMode_DIRECT_SCONV:
	  {
		  timer.Start();

		  int height = conv_input_shape_.cpu_data()[1];
		  int width = conv_input_shape_.cpu_data()[2];
		  int kernel_h = kernel_shape_.cpu_data()[0];
		  int kernel_w = kernel_shape_.cpu_data()[1];
		  int pad_h = pad_.cpu_data()[0];
		  int pad_w = pad_.cpu_data()[1];
		  int stride_h = stride_.cpu_data()[0];
		  int stride_w = stride_.cpu_data()[1];
		  int dilation_h = dilation_.cpu_data()[0];
		  int dilation_w = dilation_.cpu_data()[1];

		  const int *rowptr = nz_weight_index_pointers_.cpu_data() + row_offset * g;
		  const float *values = nz_weight_values_.cpu_data()+ weight_offset_ * g;
		  const int *colidx = nz_weight_indices_.cpu_data()+ weight_offset_ * g;

		  const int output_h = (height + 2 * pad_h -
				  (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
		  const int output_w = (width + 2 * pad_w -
				  (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
		  assert(output_h*output_w == N);
      const float *in_temp = input_padded + conv_in_channels_/group_ * g * (height + pad_h) * (width + pad_w);
      if (dilation_h != 1 || dilation_w != 1 || !this->bias_term_) {
        LOG(WARNING) << "Inefficient code path";
        for (int output_row = 0; output_row < output_h; ++output_row) {
          for (int output_col = 0; output_col < output_w; ++output_col) {

            for (int out_channel = 0; out_channel < M; ++out_channel) {
              float sum = 0;

              for (int j = rowptr[out_channel]; j < rowptr[out_channel + 1]; ++j) {
                int col = colidx[j];

                int kernel_col = col%(width + pad_w);
                int kernel_row = (col/(width + pad_w))%(height + pad_h);
                int in_channel = col/((width + pad_w)*(height + pad_h));

                int input_row = kernel_row * dilation_h + output_row * stride_h;
                int input_col = kernel_col * dilation_w + output_col * stride_w;

                sum += values[j]*in_temp[(in_channel * (height + pad_h) + input_row) * (width + pad_w) + input_col];
              }

              output[output_offset_ * g + (out_channel * output_h + output_row) * output_w + output_col] = sum;
            }
          }
			  }
		  }
		  else {
		    const float *bias = this->blobs_[1]->cpu_data();
		    const float *bias_multiplier = bias_multiplier_.cpu_data();

#if 1 //defined(__AVX2__) && defined(__INTEL_COMPILER)
        if (height == 13 && width == 13 && pad_h == 1 && pad_w == 1 && stride_h == 1 && stride_w == 1 && kernel_h == 3 && kernel_w == 3) {

          int WIDTH = 13;
          int WOUT = 13;
          int PAD = 1;

          __m256 sum[(WOUT + 1)/2][2]; // [7][2]
          __declspec(aligned(64)) float sum_temp[8];

          for (int out_channel = 0; out_channel < M; ++out_channel) {
            if (rowptr[out_channel + 1] == rowptr[out_channel]) continue;

            // Upper half of images
            int hbegin = 0, hend = (WOUT + 1)/2;
            int j = rowptr[out_channel];
            __m256 c = _mm256_set1_ps(values[j]);
            int off = colidx[j];

            for (int h = hbegin; h < hend; ++h) {
              sum[h - hbegin][0] = _mm256_mul_ps(c, _mm256_loadu_ps(in_temp + off + h*(WIDTH + PAD)));
              sum[h - hbegin][1] = _mm256_mul_ps(c, _mm256_loadu_ps(in_temp + off + h*(WIDTH + PAD) + 8));
            }

            int jbegin = rowptr[out_channel] + 1;
            int jend = rowptr[out_channel + 1];

            for (j = jbegin; j < jend; ++j) {
              c = _mm256_set1_ps(values[j]);
              off = colidx[j];

              for (int h = hbegin; h < hend; ++h) {
                sum[h - hbegin][0] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
                sum[h - hbegin][1] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
              }
            }

            for (int h = hbegin; h < hend; ++h) {
              _mm256_storeu_ps(output + output_offset_ * g  + (out_channel*WOUT + h)*WOUT, sum[h - hbegin][0]);
              _mm256_storeu_ps(sum_temp, sum[h - hbegin][1]);
              for (int w = 8; w < WOUT; ++w) {
                output[output_offset_ * g + out_channel*WOUT*WOUT + h*WOUT + w] = sum_temp[w - 8];
              }
            }

            // Lower half of images
            hbegin = (WOUT + 1)/2; hend = WOUT;
            j = rowptr[out_channel];
            c = _mm256_set1_ps(values[j]);
            off = colidx[j];

            for (int h = hbegin; h < hend; ++h) {
              sum[h - hbegin][0] = _mm256_mul_ps(c, _mm256_loadu_ps(in_temp + off + h*(WIDTH + PAD)));
              sum[h - hbegin][1] = _mm256_mul_ps(c, _mm256_loadu_ps(in_temp + off + h*(WIDTH + PAD) + 8));
            }

            for (j = jbegin; j < jend; ++j) {
              c = _mm256_set1_ps(values[j]);
              off = colidx[j];

              for (int h = hbegin; h < hend; ++h) {
                sum[h - hbegin][0] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + h*(WIDTH + PAD)), sum[h - hbegin][0]);
                sum[h - hbegin][1] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + h*(WIDTH + PAD) + 8), sum[h - hbegin][1]);
              }
            }

            for (int h = hbegin; h < hend; ++h) {
              _mm256_storeu_ps(output + output_offset_ * g + (out_channel*WOUT + h)*WOUT, sum[h - hbegin][0]);
              _mm256_storeu_ps(sum_temp, sum[h - hbegin][1]);
              for (int w = 8; w < WOUT; ++w) {
                output[output_offset_ * g + out_channel*WOUT*WOUT + h*WOUT + w] = sum_temp[w - 8];
              }
            }
          }
        }
        else if (height == 27 && width == 27 && pad_h == 2 && pad_w == 2 && stride_h == 1 && stride_w == 1 && kernel_w == 5 && kernel_h == 5) {
          int WIDTH = 27;
          int WOUT = 27;
          int PAD = 2;

          const float *bias = this->blobs_[1]->cpu_data();
          const float *bias_multiplier = bias_multiplier_.cpu_data();

          for (int out_channel = 0; out_channel < M; ++out_channel) {
            __m256 sum[(WOUT + 3)/4][2]; // [7][2]

            unsigned long long t = __rdtsc();

            int out_channel_offset = out_channel%8;

            // (0, 0) block
            int hbegin = 0, hend = (WOUT + 4)/5;
            int j;
            __m256 c;
            int off;

            int jbegin = rowptr[out_channel];
            int jend = rowptr[out_channel + 1];

            __m256 bias_v = _mm256_set1_ps(bias[out_channel]);

#undef MY_FMADD
#define MY_FMADD(HBEGIN, WBEGIN) \
            sum[0][0] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 0)*WOUT + WBEGIN)); \
            sum[0][1] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 0)*WOUT + WBEGIN + 8)); \
            sum[1][0] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 1)*WOUT + WBEGIN)); \
            sum[1][1] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 1)*WOUT + WBEGIN + 8)); \
            sum[2][0] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 2)*WOUT + WBEGIN)); \
            sum[2][1] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 2)*WOUT + WBEGIN + 8)); \
            sum[3][0] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 3)*WOUT + WBEGIN)); \
            sum[3][1] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 3)*WOUT + WBEGIN + 8)); \
            sum[4][0] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 4)*WOUT + WBEGIN)); \
            sum[4][1] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 4)*WOUT + WBEGIN + 8)); \
            sum[5][0] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 5)*WOUT + WBEGIN)); \
            sum[5][1] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 5)*WOUT + WBEGIN + 8)); \
            sum[6][0] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 6)*WOUT + WBEGIN)); \
            sum[6][1] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 6)*WOUT + WBEGIN + 8)); \
 \
            for (j = jbegin; j < jend; ++j) { \
              c = _mm256_set1_ps(values[j]); \
              off = colidx[j]; \
 \
              sum[0][0] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 0)*(WIDTH + PAD) + WBEGIN), sum[0][0]); \
              sum[0][1] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 0)*(WIDTH + PAD) + WBEGIN + 8), sum[0][1]); \
              sum[1][0] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 1)*(WIDTH + PAD) + WBEGIN), sum[1][0]); \
              sum[1][1] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 1)*(WIDTH + PAD) + WBEGIN + 8), sum[1][1]); \
              sum[2][0] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 2)*(WIDTH + PAD) + WBEGIN), sum[2][0]); \
              sum[2][1] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 2)*(WIDTH + PAD) + WBEGIN + 8), sum[2][1]); \
              sum[3][0] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 3)*(WIDTH + PAD) + WBEGIN), sum[3][0]); \
              sum[3][1] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 3)*(WIDTH + PAD) + WBEGIN + 8), sum[3][1]); \
              sum[4][0] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 4)*(WIDTH + PAD) + WBEGIN), sum[4][0]); \
              sum[4][1] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 4)*(WIDTH + PAD) + WBEGIN + 8), sum[4][1]); \
              sum[5][0] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 5)*(WIDTH + PAD) + WBEGIN), sum[5][0]); \
              sum[5][1] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 5)*(WIDTH + PAD) + WBEGIN + 8), sum[5][1]); \
              sum[6][0] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 6)*(WIDTH + PAD) + WBEGIN), sum[6][0]); \
              sum[6][1] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 6)*(WIDTH + PAD) + WBEGIN + 8), sum[6][1]); \
            } \
 \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 0)*32 + WBEGIN, sum[0][0]); \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 0)*32 + WBEGIN + 8, sum[0][1]); \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 1)*32 + WBEGIN, sum[1][0]); \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 1)*32 + WBEGIN + 8, sum[1][1]); \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 2)*32 + WBEGIN, sum[2][0]); \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 2)*32 + WBEGIN + 8, sum[2][1]); \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 3)*32 + WBEGIN, sum[3][0]); \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 3)*32 + WBEGIN + 8, sum[3][1]); \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 4)*32 + WBEGIN, sum[4][0]); \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 4)*32 + WBEGIN + 8, sum[4][1]); \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 5)*32 + WBEGIN, sum[5][0]); \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 5)*32 + WBEGIN + 8, sum[5][1]); \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 6)*32 + WBEGIN, sum[6][0]); \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 6)*32 + WBEGIN + 8, sum[6][1]);

            MY_FMADD(0, 0);
            MY_FMADD(0, 16);

            MY_FMADD(7, 0);
            MY_FMADD(7, 16);

            MY_FMADD(14, 0);
            MY_FMADD(14, 16);

#undef MY_FMADD
#define MY_FMADD(HBEGIN, WBEGIN) \
            sum[0][0] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 0)*WOUT + WBEGIN)); \
            sum[0][1] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 0)*WOUT + WBEGIN + 8)); \
            sum[1][0] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 1)*WOUT + WBEGIN)); \
            sum[1][1] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 1)*WOUT + WBEGIN + 8)); \
            sum[2][0] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 2)*WOUT + WBEGIN)); \
            sum[2][1] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 2)*WOUT + WBEGIN + 8)); \
            sum[3][0] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 3)*WOUT + WBEGIN)); \
            sum[3][1] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 3)*WOUT + WBEGIN + 8)); \
            sum[4][0] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 4)*WOUT + WBEGIN)); \
            sum[4][1] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 4)*WOUT + WBEGIN + 8)); \
            sum[5][0] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 5)*WOUT + WBEGIN)); \
            sum[5][1] = _mm256_mul_ps(bias_v, _mm256_loadu_ps(bias_multiplier + (HBEGIN + 5)*WOUT + WBEGIN + 8)); \
 \
            for (j = jbegin; j < jend; ++j) { \
              c = _mm256_set1_ps(values[j]); \
              off = colidx[j]; \
 \
              sum[0][0] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 0)*(WIDTH + PAD) + WBEGIN), sum[0][0]); \
              sum[0][1] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 0)*(WIDTH + PAD) + WBEGIN + 8), sum[0][1]); \
              sum[1][0] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 1)*(WIDTH + PAD) + WBEGIN), sum[1][0]); \
              sum[1][1] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 1)*(WIDTH + PAD) + WBEGIN + 8), sum[1][1]); \
              sum[2][0] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 2)*(WIDTH + PAD) + WBEGIN), sum[2][0]); \
              sum[2][1] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 2)*(WIDTH + PAD) + WBEGIN + 8), sum[2][1]); \
              sum[3][0] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 3)*(WIDTH + PAD) + WBEGIN), sum[3][0]); \
              sum[3][1] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 3)*(WIDTH + PAD) + WBEGIN + 8), sum[3][1]); \
              sum[4][0] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 4)*(WIDTH + PAD) + WBEGIN), sum[4][0]); \
              sum[4][1] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 4)*(WIDTH + PAD) + WBEGIN + 8), sum[4][1]); \
              sum[5][0] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 5)*(WIDTH + PAD) + WBEGIN), sum[5][0]); \
              sum[5][1] = _mm256_fmadd_ps(c, _mm256_loadu_ps(in_temp + off + (HBEGIN + 5)*(WIDTH + PAD) + WBEGIN + 8), sum[5][1]); \
            } \
 \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 0)*32 + WBEGIN, sum[0][0]); \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 0)*32 + WBEGIN + 8, sum[0][1]); \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 1)*32 + WBEGIN, sum[1][0]); \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 1)*32 + WBEGIN + 8, sum[1][1]); \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 2)*32 + WBEGIN, sum[2][0]); \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 2)*32 + WBEGIN + 8, sum[2][1]); \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 3)*32 + WBEGIN, sum[3][0]); \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 3)*32 + WBEGIN + 8, sum[3][1]); \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 4)*32 + WBEGIN, sum[4][0]); \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 4)*32 + WBEGIN + 8, sum[4][1]); \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 5)*32 + WBEGIN, sum[5][0]); \
            _mm256_store_ps(output + (out_channel_offset*WOUT + HBEGIN + 5)*32 + WBEGIN + 8, sum[5][1]);

            MY_FMADD(21, 0);
            MY_FMADD(21, 16);
#undef MY_FMADD

            if (0 == omp_get_thread_num()) conv_time += __rdtsc() - t;

            if (out_channel%8 != 7) continue;

            t = __rdtsc();

            // transpose to vectorize pooling layer over multiple channels
            for (int h = 0; h < WOUT; ++h) {
              for (int w = 0; w < WOUT/8*8; w += 8) {
                __m256 v0 = _mm256_load_ps(output + h*32 + w);
                __m256 v1 = _mm256_load_ps(output + (WOUT + h)*32 + w);
                __m256 v2 = _mm256_load_ps(output + (2*WOUT + h)*32 + w);
                __m256 v3 = _mm256_load_ps(output + (3*WOUT + h)*32 + w);
                __m256 v4 = _mm256_load_ps(output + (4*WOUT + h)*32 + w);
                __m256 v5 = _mm256_load_ps(output + (5*WOUT + h)*32 + w);
                __m256 v6 = _mm256_load_ps(output + (6*WOUT + h)*32 + w);
                __m256 v7 = _mm256_load_ps(output + (7*WOUT + h)*32 + w);

                transpose8_ps(v0, v1, v2, v3, v4, v5, v6, v7);

                _mm256_store_ps(output + ((32 + h)*WOUT + w)*8, v0);
                _mm256_store_ps(output + ((32 + h)*WOUT + (w + 1))*8, v1);
                _mm256_store_ps(output + ((32 + h)*WOUT + (w + 2))*8, v2);
                _mm256_store_ps(output + ((32 + h)*WOUT + (w + 3))*8, v3);
                _mm256_store_ps(output + ((32 + h)*WOUT + (w + 4))*8, v4);
                _mm256_store_ps(output + ((32 + h)*WOUT + (w + 5))*8, v5);
                _mm256_store_ps(output + ((32 + h)*WOUT + (w + 6))*8, v6);
                _mm256_store_ps(output + ((32 + h)*WOUT + (w + 7))*8, v7);
              }
              for (int w = WOUT/8*8; w < WOUT; ++w) {
                for (int i = 0; i < 8; ++i) {
                  output[((32 + h)*WOUT + w)*8 + i] = output[(i*WOUT + h)*32 + w];
                }
              }
            }

            if (0 == omp_get_thread_num()) transpose_time += __rdtsc() - t;
            t = __rdtsc();

            float *pool_top = ((ConvolutionReLUPoolLRNLayer<float> *)this)->pool_top_[0]->mutable_cpu_data() + ((ConvolutionReLUPoolLRNLayer<float> *)this)->pool_top_[0]->offset(batch_idx);
            int *mask = ((ConvolutionReLUPoolLRNLayer<float> *)this)->max_idx_.mutable_cpu_data() + ((ConvolutionReLUPoolLRNLayer<float> *)this)->pool_top_[0]->offset(batch_idx);

            const int STRIDE_POOL = 2;
            const int K_POOL = 3;
            const int POOLED_WIDTH = (WOUT - K_POOL + STRIDE_POOL - 1) / STRIDE_POOL + 1; // (27 - 3 + 1)/2 + 1 = 13

            const float *conv_top_data_cur = output + 8*WOUT*32;
            float *pool_top_data_cur = pool_top + (M*g + out_channel - 7)*POOLED_WIDTH*POOLED_WIDTH;
            int *mask_cur = mask + (M*g + out_channel - 7)*POOLED_WIDTH*POOLED_WIDTH;

            __declspec(aligned(64)) float maximum[8];

            __declspec(aligned(64)) int identity[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
            __m256i identity_v = _mm256_load_si256((__m256i *)identity);

            for (int ph = 0; ph < POOLED_WIDTH; ++ph) {
              __declspec(aligned(64)) int mask[8];

              int hstart = ph * STRIDE_POOL;
              int hend = hstart + K_POOL;

              for (int pw = 0; pw < POOLED_WIDTH; ++pw) {
                int wstart = pw * STRIDE_POOL;
                __m256 maximum_v = _mm256_setzero_ps(); // JSP: using 0 instead of -FLT_MAX does ReLU for us.
                __m256 mask_v = _mm256_setzero_ps();
                __m256 cmp_v, in_v;

                int index = hstart * WOUT + wstart;
                in_v = _mm256_load_ps(conv_top_data_cur + index*8);
                cmp_v = _mm256_cmp_ps(in_v, maximum_v, _CMP_LE_OQ);
                maximum_v =  _mm256_blendv_ps(in_v, maximum_v, cmp_v);
                mask_v = _mm256_blendv_ps(
                    _mm256_castsi256_ps(_mm256_add_epi32(_mm256_set1_epi32(index*8), identity_v)),
                    mask_v,
                    cmp_v);

                index = hstart * WOUT + wstart + 1;
                in_v = _mm256_load_ps(conv_top_data_cur + index*8);
                cmp_v = _mm256_cmp_ps(in_v, maximum_v, _CMP_LE_OQ);
                maximum_v =  _mm256_blendv_ps(in_v, maximum_v, cmp_v);
                mask_v = _mm256_blendv_ps(
                    _mm256_castsi256_ps(_mm256_add_epi32(_mm256_set1_epi32(index*8), identity_v)),
                    mask_v,
                    cmp_v);

                index = hstart * WOUT + wstart + 2;
                in_v = _mm256_load_ps(conv_top_data_cur + index*8);
                cmp_v = _mm256_cmp_ps(in_v, maximum_v, _CMP_LE_OQ);
                maximum_v =  _mm256_blendv_ps(in_v, maximum_v, cmp_v);
                mask_v = _mm256_blendv_ps(
                    _mm256_castsi256_ps(_mm256_add_epi32(_mm256_set1_epi32(index*8), identity_v)),
                    mask_v,
                    cmp_v);

                index = (hstart + 1) * WOUT + wstart;
                in_v = _mm256_load_ps(conv_top_data_cur + index*8);
                cmp_v = _mm256_cmp_ps(in_v, maximum_v, _CMP_LE_OQ);
                maximum_v =  _mm256_blendv_ps(in_v, maximum_v, cmp_v);
                mask_v = _mm256_blendv_ps(
                    _mm256_castsi256_ps(_mm256_add_epi32(_mm256_set1_epi32(index*8), identity_v)),
                    mask_v,
                    cmp_v);

                index = (hstart + 1) * WOUT + wstart + 1;
                in_v = _mm256_load_ps(conv_top_data_cur + index*8);
                cmp_v = _mm256_cmp_ps(in_v, maximum_v, _CMP_LE_OQ);
                maximum_v =  _mm256_blendv_ps(in_v, maximum_v, cmp_v);
                mask_v = _mm256_blendv_ps(
                    _mm256_castsi256_ps(_mm256_add_epi32(_mm256_set1_epi32(index*8), identity_v)),
                    mask_v,
                    cmp_v);

                index = (hstart + 1) * WOUT + wstart + 2;
                in_v = _mm256_load_ps(conv_top_data_cur + index*8);
                cmp_v = _mm256_cmp_ps(in_v, maximum_v, _CMP_LE_OQ);
                maximum_v =  _mm256_blendv_ps(in_v, maximum_v, cmp_v);
                mask_v = _mm256_blendv_ps(
                    _mm256_castsi256_ps(_mm256_add_epi32(_mm256_set1_epi32(index*8), identity_v)),
                    mask_v,
                    cmp_v);

                index = (hstart + 2) * WOUT + wstart;
                in_v = _mm256_load_ps(conv_top_data_cur + index*8);
                cmp_v = _mm256_cmp_ps(in_v, maximum_v, _CMP_LE_OQ);
                maximum_v =  _mm256_blendv_ps(in_v, maximum_v, cmp_v);
                mask_v = _mm256_blendv_ps(
                    _mm256_castsi256_ps(_mm256_add_epi32(_mm256_set1_epi32(index*8), identity_v)),
                    mask_v,
                    cmp_v);

                index = (hstart + 2) * WOUT + wstart + 1;
                in_v = _mm256_load_ps(conv_top_data_cur + index*8);
                cmp_v = _mm256_cmp_ps(in_v, maximum_v, _CMP_LE_OQ);
                maximum_v =  _mm256_blendv_ps(in_v, maximum_v, cmp_v);
                mask_v = _mm256_blendv_ps(
                    _mm256_castsi256_ps(_mm256_add_epi32(_mm256_set1_epi32(index*8), identity_v)),
                    mask_v,
                    cmp_v);

                index = (hstart + 2) * WOUT + wstart + 2;
                in_v = _mm256_load_ps(conv_top_data_cur + index*8);
                cmp_v = _mm256_cmp_ps(in_v, maximum_v, _CMP_LE_OQ);
                maximum_v =  _mm256_blendv_ps(in_v, maximum_v, cmp_v);
                mask_v = _mm256_blendv_ps(
                    _mm256_castsi256_ps(_mm256_add_epi32(_mm256_set1_epi32(index*8), identity_v)),
                    mask_v,
                    cmp_v);

                _mm256_store_ps(maximum, maximum_v);
                _mm256_store_ps((float *)mask, mask_v);

                const int pool_index = ph * POOLED_WIDTH + pw;
                for (int j = 0; j < 8; ++j) {
                  pool_top_data_cur[pool_index + j*POOLED_WIDTH*POOLED_WIDTH] = maximum[j];
                  mask_cur[pool_index + j*POOLED_WIDTH*POOLED_WIDTH] = mask[j];
                }
              }
            }

            if (0 == omp_get_thread_num()) pool_time += __rdtsc() - t;
          } // for each out channel
        }
        else
#endif
        if (height == 227 && width == 227 && pad_h == 0 && pad_w == 0 && stride_h == 4 && stride_w == 4 && kernel_w == 11 && kernel_h == 11) {
          int WIDTH = 227;
          int STRIDE = 4;
          int K = 11;
          int WOUT = (WIDTH - K)/STRIDE + 1; // 55
          const int JBLOCK = 128;
          const int HBLOCK = 8;
          const int WBLOCK = 9;

          __declspec(aligned(64)) float sum[WOUT*WOUT];

          for (int out_channel = 0; out_channel < M; ++out_channel) {
            int jbegin = rowptr[out_channel];
            int jend = std::min(jbegin + JBLOCK, rowptr[out_channel + 1]);

            for (int hbegin = 0; hbegin < WOUT; hbegin += HBLOCK) {
              int hend = std::min(hbegin + HBLOCK, WOUT);

              for (int wbegin = 0; wbegin < WOUT; wbegin += WBLOCK) {
                int wend = std::min(wbegin + WBLOCK, WOUT);

                for (int k = 0; k < (hend - hbegin) * (wend - wbegin); ++k) {
                  sum[k] = 0;
                }

                for (int j = jbegin; j < jend; ++j) {
                  float c = values[j];
                  int off = colidx[j];
                  int k = 0;
                  for (int h = hbegin; h < hend; ++h) {
                    for (int w = wbegin; w < wend; ++w, ++k) {
                      sum[k] += c*in_temp[off + (h*WIDTH + w)*STRIDE];
                    }
                  }
                }

                int k = 0;
                for (int h = hbegin; h < hend; ++h) {
                  for (int w = wbegin; w < wend; ++w, ++k) {
                    output[output_offset_ * g + (out_channel*WOUT + h)*WOUT + w] = sum[k];
                  }
                }
              }
            }
            jbegin += JBLOCK;

            for ( ; jbegin < rowptr[out_channel + 1]; jbegin += JBLOCK) {
              int jend = std::min(jbegin + JBLOCK, rowptr[out_channel + 1]);

              for (int hbegin = 0; hbegin < WOUT; hbegin += HBLOCK) {
                int hend = std::min(hbegin + HBLOCK, WOUT);

                for (int wbegin = 0; wbegin < WOUT; wbegin += WBLOCK) {
                  int wend = std::min(wbegin + WBLOCK, WOUT);

                  for (int k = 0; k < (hend - hbegin) * (wend - wbegin); ++k) {
                    sum[k] = 0;
                  }

                  for (int j = jbegin; j < jend; ++j) {
                    float c = values[j];
                    int off = colidx[j];
                    int k = 0;
                    for (int h = hbegin; h < hend; ++h) {
                      for (int w = wbegin; w < wend; ++w, ++k) {
                        sum[k] += c*in_temp[off + (h*WIDTH + w)*STRIDE];
                      }
                    }
                  }

                  int k = 0;
                  for (int h = hbegin; h < hend; ++h) {
                    for (int w = wbegin; w < wend; ++w, ++k) {
                      output[output_offset_ * g + (out_channel*WOUT + h)*WOUT + w] += sum[k];
                    }
                  }
                }
              }
            }

//            for (int j = rowptr[out_channel]; j < rowptr[out_channel + 1]; ++j) {
//              float c = values[j];
//              int off = colidx[j];
//              for (int h = 0; h < WOUT/2; ++h) {
//                for (int w = WOUT/2; w < WOUT; ++w) {
//                  sum[h*WOUT + w] += c*in_temp[off + (h*WIDTH + w)*STRIDE];
//                }
//              }
//            }
//
//            for (int h = 0; h < WOUT/2; ++h) {
//              for (int w = WOUT/2; w < WOUT; ++w) {
//                output[output_offset_ * g + (out_channel*WOUT + h)*WOUT + w] = sum[h*WOUT + w];
//              }
//            }
//
//            for (int h = WOUT/2; h < WOUT; ++h) {
//              for (int w = 0; w < WOUT/2; ++w) {
//                sum[h*WOUT + w] = 0;
//              }
//            }
//
//            for (int j = rowptr[out_channel]; j < rowptr[out_channel + 1]; ++j) {
//              float c = values[j];
//              int off = colidx[j];
//              for (int h = WOUT/2; h < WOUT; ++h) {
//                for (int w = 0; w < WOUT/2; ++w) {
//                  sum[h*WOUT + w] += c*in_temp[off + (h*WIDTH + w)*STRIDE];
//                }
//              }
//            }
//
//            for (int h = WOUT/2; h < WOUT; ++h) {
//              for (int w = 0; w < WOUT/2; ++w) {
//                output[output_offset_ * g + (out_channel*WOUT + h)*WOUT + w] = sum[h*WOUT + w];
//              }
//            }
//
//            for (int h = WOUT/2; h < WOUT; ++h) {
//              for (int w = WOUT/2; w < WOUT; ++w) {
//                sum[h*WOUT + w] = 0;
//              }
//            }
//
//            for (int j = rowptr[out_channel]; j < rowptr[out_channel + 1]; ++j) {
//              float c = values[j];
//              int off = colidx[j];
//              for (int h = WOUT/2; h < WOUT; ++h) {
//                for (int w = WOUT/2; w < WOUT; ++w) {
//                  sum[h*WOUT + w] += c*in_temp[off + (h*WIDTH + w)*STRIDE];
//                }
//              }
//            }
//
//            for (int h = WOUT/2; h < WOUT; ++h) {
//              for (int w = WOUT/2; w < WOUT; ++w) {
//                output[output_offset_ * g + (out_channel*WOUT + h)*WOUT + w] = sum[h*WOUT + w];
//              }
//            }
          }
        }
        else
        {
          for (int output_row = 0; output_row < output_h; ++output_row) {
            for (int output_col = 0; output_col < output_w; ++output_col) {

              const float *in_temp2 = in_temp + output_row * stride_h * (width + pad_w) + output_col * stride_w;

              for (int out_channel = 0; out_channel < M; ++out_channel) {
                float sum = 0;

                for (int j = rowptr[out_channel]; j < rowptr[out_channel + 1]; ++j) {
                  assert(in_temp2 + colidx[j] - input_padded < input_padded_len);
                  sum += values[j]*in_temp2[colidx[j]];
                }

                output[output_offset_ * g + (out_channel*output_h + output_row)*output_w + output_col] = sum;
              }
            }
          } // !__AVX2__
				}
		  }
		  break;
	  }
	  default:
		timer.Start();
		caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, M, N, K,
				  (float)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
				  (float)0., output + output_offset_ * g);
		break;
	  }
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  if(conv_param.conv_mode() != caffe::ConvolutionParameter_ConvMode_LOWERED_GEMM){
	  LOG(FATAL)<<"Training in sparse format is not supported";
  }
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  if(conv_param.conv_mode() != caffe::ConvolutionParameter_ConvMode_LOWERED_GEMM){
	  LOG(FATAL)<<"Training in sparse format is not supported";
  }
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }

  for (int g = 0; g < group_; ++g) {
#ifdef	GPU_USE_CUSPARSE
	  int total_nonzero = 0;
	  caffe_gpu_sparse_dense2csr(kernel_dim_ / group_, conv_out_spatial_dim_,
						  col_buff + col_offset_ * g,
						  nonzero_per_rowcol_buffer_.mutable_gpu_data(),
						  nonzero_elements_buffer_.mutable_gpu_data(),
						  index_pointers_buffer_.mutable_gpu_data(),
						  nonzero_indices_buffer_.mutable_gpu_data(), &total_nonzero);
	  Dtype sparsity = (Dtype)1.0 - (Dtype)total_nonzero/(Dtype)(kernel_dim_*height_out_*width_out_);
	  //LOG(INFO)<<"Sparsity of "<< Layer<Dtype>::layer_param().name() << ": "<< sparsity;
	  if(sparsity<(Dtype)0.9){
#endif
		 caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
			 (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
			 (Dtype)0., output + output_offset_ * g);
#ifdef	GPU_USE_CUSPARSE
	  }else{
		 //dense weight matrix multi. sparse feature map matrix
		 //WARNING WARNING WARNING: When A*B, B in format CSR is slow
		 caffe_gpu_sparse_mmcsr(conv_out_channels_ /group_, conv_out_spatial_dim_, kernel_dim_ / group_,
				  (Dtype)1., weights + weight_offset_ * g,
				  total_nonzero,
				  nonzero_elements_buffer_.gpu_data(),
				  index_pointers_buffer_.gpu_data(),
				  nonzero_indices_buffer_.gpu_data(),
				  (Dtype)0., output + output_offset_ * g);
	  }
#endif
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  if(conv_param.conv_mode() != caffe::ConvolutionParameter_ConvMode_LOWERED_GEMM){
	  LOG(FATAL)<<"Training in sparse format is not supported";
  }
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  if(conv_param.conv_mode() != caffe::ConvolutionParameter_ConvMode_LOWERED_GEMM){
	  LOG(FATAL)<<"Training in sparse format is not supported";
  }
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
