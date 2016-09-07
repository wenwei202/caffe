#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/mmio.hpp"

namespace caffe {

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
	int left_cols = 0;
	int left_rows = 0;
	switch(conv_param.conv_mode()){
		case caffe::ConvolutionParameter_ConvMode_LOWERED_CSRMM:
			LOG(INFO)<<"ConvolutionParameter_ConvMode_LOWERED_CSRMM";
			for (int g = 0; g < group_; ++g) {
				switch (Caffe::mode()) {
				    case Caffe::CPU:
				    	caffe_cpu_sparse_dense2csr(M, N,
								this->blobs_[0]->mutable_cpu_data() + weight_offset * g,
								nz_weight_values_.mutable_cpu_data()+ weight_offset * g,
								nz_weight_indices_.mutable_cpu_data()+ weight_offset * g,
								nz_weight_index_pointers_.mutable_cpu_data() + row_offset * g);
				      break;
				    case Caffe::GPU:{
				#ifndef CPU_ONLY
				    	int total_nonzero = 0;
				    	caffe_gpu_sparse_dense2csr(M, N,
				    			this->blobs_[0]->gpu_data() + weight_offset * g,
				    			nz_per_row_.mutable_gpu_data() + M*g,
							    nz_weight_values_.mutable_gpu_data()+ weight_offset * g,
							    nz_weight_index_pointers_.mutable_gpu_data() + row_offset * g,
							    nz_weight_indices_.mutable_gpu_data()+ weight_offset * g,
							    &total_nonzero);
				    	nz_num_[g] = total_nonzero;
				#else
				      NO_GPU;
				#endif
				      break;
				    }
				}

			}
			break;
		case caffe::ConvolutionParameter_ConvMode_LOWERED_CCNMM:{
			is_concatenating_weights_features_ = true;
			LOG(INFO)<<"ConvolutionParameter_ConvMode_LOWERED_CCNMM";

			//analyze column sparsity
			for (int g = 0; g < group_; ++g) {
				caffe_cpu_if_all_zero(this->blobs_[0]->shape(0)/group_,
						this->blobs_[0]->count(1,4),
						this->blobs_[0]->cpu_data() + this->blobs_[0]->count()/group_ * g,
						col_buf_mask_.mutable_cpu_data() + this->blobs_[0]->count(1,4) * g);
			}
			//analyze row sparsity
			caffe_cpu_if_all_zero(this->blobs_[0]->shape(0),
									this->blobs_[0]->count(1,4),
									this->blobs_[0]->cpu_data(),
									row_buf_mask_.mutable_cpu_data(),
									false);

			// concatenating weight matrix
			left_columns_.erase(left_columns_.begin(),left_columns_.end());
			left_rows_.erase(left_rows_.begin(),left_rows_.end());
			LOG(INFO)<<"concatenating weight matrix";
			int total_weights = 0;
			for (int g = 0; g < group_; ++g) {
				left_cols = kernel_dim_ - caffe_cpu_asum( kernel_dim_, col_buf_mask_.cpu_data()+ kernel_dim_ * g);
				left_columns_.push_back(left_cols);
				left_rows = conv_out_channels_ /group_ - caffe_cpu_asum( conv_out_channels_ /group_, row_buf_mask_.cpu_data()+ conv_out_channels_ /group_ * g);
				left_rows_.push_back(left_rows);
				total_weights += left_cols*left_rows;
				LOG(INFO)<<layerparam.name()<<" left_cols="<<left_cols<<" left_rows="<<left_rows;
			}
			squeezed_weight_buffer_.Reshape(1,1,1,total_weights);
			LOG(INFO)<<"squeezing weight matrix";
			int weight_offset_sum = 0;
			for (int g = 0; g < group_; ++g) {
//				caffe_cpu_del_zero_cols(conv_out_channels_ /group_,
//					  kernel_dim_ ,
//					  this->blobs_[0]->cpu_data() + weight_offset_ * g,
//					  squeezed_weight_buffer_.mutable_cpu_data() + weight_offset_ * g,
//					  &left_cols,
//					  col_buf_mask_.cpu_data() + kernel_dim_ * g );
//				left_columns_.push_back(left_cols);
				//squeezed_weight_groups_[g].reset(new Blob<Dtype>(1,1,left_rows_[g],left_columns_[g]));
				LOG(INFO)<<layerparam.name()<<" squeezing to "<<left_rows_[g]<<"x"<<left_columns_[g];
				caffe_cpu_concatenate_rows_cols(
						conv_out_channels_ /group_,
						kernel_dim_,
						this->blobs_[0]->cpu_data() + weight_offset_ * g,
						//squeezed_weight_groups_[g]->mutable_cpu_data(),
						squeezed_weight_buffer_.mutable_cpu_data() + weight_offset_sum,
						col_buf_mask_.cpu_data()+ kernel_dim_ * g,
						row_buf_mask_.cpu_data()+ conv_out_channels_ /group_ * g
						);
				weight_offset_sum += left_rows_[g]*left_columns_[g];
			}
			LOG(INFO)<<"weight matrix squeezed";
			break;
		}
		case caffe::ConvolutionParameter_ConvMode_DIRECT_SCONV:
			{
				LOG(INFO)<<"ConvolutionParameter_ConvMode_DIRECT_SCONV";
				for (int g = 0; g < group_; ++g) {
					// first create a CSR matrix as for LOWERED_CSRMM
					caffe_cpu_sparse_dense2csr(M, N,
							this->blobs_[0]->mutable_cpu_data() + weight_offset * g,
							nz_weight_values_.mutable_cpu_data()+ weight_offset * g,
							nz_weight_indices_.mutable_cpu_data()+ weight_offset * g,
							nz_weight_index_pointers_.mutable_cpu_data() + row_offset * g);

					int height = conv_input_shape_.cpu_data()[1];
					int width = conv_input_shape_.cpu_data()[2];
					int kernel_h = kernel_shape_.cpu_data()[0];
					int kernel_w = kernel_shape_.cpu_data()[1];
					int pad_h = pad_.cpu_data()[0];
					int pad_w = pad_.cpu_data()[1];

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

					printf("k-mode sparsity %g\n", (double)num_of_non_zero_kernels/(M*conv_in_channels_));
				}
				break;
			}
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
  //squeezed_weight_buffer_.Reshape(this->blobs_[0]->shape(0),this->blobs_[0]->shape(1),this->blobs_[0]->shape(2),this->blobs_[0]->shape(3));
  //squeezed_weight_groups_.resize(group_);

#ifdef USE_SNAPSHOT_FEATURE
  num_forward_image_ = 0;
#endif
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
  if(Caffe::mode()==Caffe::CPU){
	  col_buffer_shape_.push_back(kernel_dim_ * group_ * num_);
  }else{
	  col_buffer_shape_.push_back(kernel_dim_ * group_);
  }
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
	  row_buf_mask_.Reshape(1,1,1,conv_out_channels_);
#ifdef	GPU_USE_CUSPARSE
	  nonzero_elements_buffer_.Reshape(1, 1, 1, col_buffer_.count());//WARNING: real sparse matrix needs many less memory
	  nonzero_indices_buffer_.Reshape(1,1,1,nonzero_elements_buffer_.count());
	  index_pointers_buffer_.Reshape(1,1,1,col_buffer_.shape(1)+1);
	  nonzero_per_rowcol_buffer_.Reshape(1,1,1,col_buffer_.shape(1));
#endif
	  nz_weight_values_.Reshape(1, 1, 1, this->blobs_[0]->count());//nonzero elements
	  nz_weight_indices_.Reshape(1,1,1,nz_weight_values_.count());//index of nonzero
	  nz_weight_index_pointers_.Reshape(1,1,1,this->blobs_[0]->shape(0)+group_);//pointer(index) of indices
	  nz_per_row_.Reshape(1,1,1,this->blobs_[0]->shape(0));
	  nz_num_.resize(group_);
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
  transposed_output_buffer_.Reshape(1,1,conv_out_spatial_dim_,conv_out_channels_/group_);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, int batch_idx, bool skip_im2col) {
  const Dtype* col_buff = input;

  Timer timer;
  timer.Start();

  Dtype *input_padded;
  int input_padded_len;
  if (this->layer_param_.convolution_param().conv_mode() == caffe::ConvolutionParameter_ConvMode_DIRECT_SCONV) {
	  int height = conv_input_shape_.cpu_data()[1];
	  int width = conv_input_shape_.cpu_data()[2];
	  int pad_h = pad_.cpu_data()[0];
	  int pad_w = pad_.cpu_data()[1];

	  input_padded_len = conv_in_channels_ * (height + pad_h) * (width + pad_w) + pad_h * (width + 2 * pad_w);
	  input_padded = new Dtype[input_padded_len];
	  assert(input_padded);
	  for (int in_channel = 0; in_channel < conv_in_channels_; ++in_channel) {
	    memset(
	        input_padded + in_channel * (height + pad_h) * (width + pad_w),
	        0, sizeof(Dtype) * pad_h * (width + pad_w));
	  	for (int input_row = 0; input_row < height; ++input_row) {
	  	  memset(
	  	      input_padded + (in_channel * (height + pad_h) + input_row + pad_h) * (width + pad_w),
	  	      0, sizeof(Dtype) * pad_w);
	  	  memcpy(
	  	      input_padded + (in_channel * (height + pad_h) + input_row + pad_h) * (width + pad_w) + pad_w,
	  	      input + (in_channel * height + input_row) * width,
	  	      sizeof(Dtype) * width);
	  	}
	  }
	  memset(
	      input_padded + conv_in_channels_ * (height + pad_h) * (width + pad_w),
	      0,
	      sizeof(Dtype) * pad_h * (width + 2 * pad_w));
  }
  else if (!is_1x1_ ||  is_concatenating_weights_features_) {
    int offset = 0;
    if (!skip_im2col || is_concatenating_weights_features_) {
      offset = col_offset_*group_*batch_idx;
      Dtype *col_buff_mutable = col_buffer_.mutable_cpu_data() + offset;
      if(is_concatenating_weights_features_){
    	  conv_im2col_cpu(input, col_buff_mutable, col_buf_mask_.mutable_cpu_data()/*, dense_feature_map_mask_.mutable_cpu_data()*/);
      }else{
    	  conv_im2col_cpu(input, col_buff_mutable);
      }
    }
    col_buff = col_buffer_.cpu_data() + offset;
  }

  int col_buf_offset_sum = 0;
  int output_offset_sum = 0;
  int weight_offset_sum = 0;
  Timer total_timer;
  total_timer.Start();
  const int M = conv_out_channels_ /group_;
  const int N = conv_out_spatial_dim_;
  const int K = kernel_dim_;
  for (int g = 0; g < group_; ++g) {
	  const int row_offset = conv_out_channels_ /group_ + 1;
	  switch(this->layer_param_.convolution_param().conv_mode()){
	  case caffe::ConvolutionParameter_ConvMode_LOWERED_CSRMM :
		  timer.Start();
		  caffe_cpu_sparse_mmcsr(M,
				  N,
				  K,
				  (Dtype)1.,
				  nz_weight_values_.cpu_data()+ weight_offset_ * g,
				  nz_weight_indices_.cpu_data()+ weight_offset_ * g,
				  nz_weight_index_pointers_.cpu_data() + row_offset * g,
				  nz_weight_index_pointers_.cpu_data() + row_offset * g + 1,
				  col_buff + col_offset_ * g,
				  (Dtype)0.,output + output_offset_ * g);
#ifdef USE_PROFILE_DISPLAY
		  LOG(INFO)<<this->layer_param().name()<<"\t group "<<g<<": "<<timer.MicroSeconds()<<" us (Compressed Row Storage Timing)";
#endif
		  break;
	  case caffe::ConvolutionParameter_ConvMode_LOWERED_CCNMM :{
		  //LOG(INFO)<<"Computing ConvolutionParameter_ConvMode_LOWERED_CCNMM";
		  int left_cols = left_columns_[g];
		  int left_rows = left_rows_[g];
		  timer.Start();
//		  caffe_cpu_cblas_gemm(conv_out_channels_ /
//				  group_, conv_out_spatial_dim_, left_cols,
//				  (Dtype)1., squeezed_weight_buffer_.cpu_data() + weight_offset_ * g,
//				  kernel_dim_ , col_buff + col_buf_offset_sum,
//				conv_out_spatial_dim_, (Dtype)0., output + output_offset_ * g, conv_out_spatial_dim_);
//		  col_buf_offset_sum += left_cols * conv_out_spatial_dim_;
		  caffe_cpu_cblas_gemm(left_rows, conv_out_spatial_dim_, left_cols,
				  //(Dtype)1., squeezed_weight_groups_[g]->cpu_data(),
				  (Dtype)1., squeezed_weight_buffer_.cpu_data() + weight_offset_sum,
				  left_cols , col_buff + col_buf_offset_sum,
				conv_out_spatial_dim_, (Dtype)0., output + output_offset_sum, conv_out_spatial_dim_);
#ifdef USE_PROFILE_DISPLAY
		  LOG(INFO)<<this->layer_param().name()<<"\t group "<<g<<": "<<timer.MicroSeconds()<<" us (Concatenation Timing)";
#endif
		  col_buf_offset_sum += left_cols * conv_out_spatial_dim_;
		  output_offset_sum += left_rows * conv_out_spatial_dim_;
		  weight_offset_sum += left_rows*left_cols;
		  //dispatch output feature maps
		  if(group_-1 == g){
			  //LOG(INFO)<<"dispatching output feature maps";
			  caffe_cpu_dispatch_rows(conv_out_channels_,conv_out_spatial_dim_,output,row_buf_mask_.cpu_data());
			  //LOG(INFO)<<"output feature maps dispatched";
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

		  int begin = 0;
		  int end = M;

		  const int *rowptr = nz_weight_index_pointers_.cpu_data() + row_offset * g;
		  const Dtype *values = nz_weight_values_.cpu_data()+ weight_offset_ * g;
		  const int *colidx = nz_weight_indices_.cpu_data()+ weight_offset_ * g;

		  const int output_h = (height + 2 * pad_h -
				  (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
		  const int output_w = (width + 2 * pad_w -
				  (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
		  assert(output_h*output_w == N);
	      const Dtype *in_temp = input_padded + conv_in_channels_/group_ * g * (height + pad_h) * (width + pad_w);
	      if (dilation_h != 1 || dilation_w != 1) {
			  for (int output_row = 0; output_row < output_h; ++output_row) {
				for (int output_col = 0; output_col < output_w; ++output_col) {

				  for (int out_channel = begin; out_channel < end; ++out_channel) {
					Dtype sum = 0;

					for (int j = rowptr[out_channel]; j < rowptr[out_channel + 1]; ++j) {
						int col = colidx[j];

						int kernel_col = col%(width + pad_w);
						int kernel_row = (col/(width + pad_w))%(height + pad_h);
						int in_channel = col/((width + pad_w)*(height + pad_h));

						int input_row = kernel_row * dilation_h + output_row * stride_h;
						int input_col = kernel_col * dilation_w + output_col * stride_w;

						sum += values[j]*in_temp[(in_channel * (height + pad_h) + input_row) * (width + pad_w) + input_col];
					}

					output[output_offset_ * g + (out_channel*output_h + output_row)*output_w + output_col] = sum;
				  }
				}
			  }
		  }
		  else {
			  for (int output_row = 0; output_row < output_h; ++output_row) {
					for (int output_col = 0; output_col < output_w; ++output_col) {

					  const Dtype *in_temp2 = in_temp + output_row * stride_h * (width + pad_w) + output_col * stride_w;

					  for (int out_channel = begin; out_channel < end; ++out_channel) {
              Dtype sum = 0;

              for (int j = rowptr[out_channel]; j < rowptr[out_channel + 1]; ++j) {
                assert(in_temp2 + colidx[j] - input_padded < input_padded_len);
                sum += values[j]*in_temp2[colidx[j]];
              }

              output[output_offset_ * g + (out_channel*output_h + output_row)*output_w + output_col] = sum;
					  }
					}
				}
		  }
		  break;
	  }
	  default:
		timer.Start();
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, N, K,
				  (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
				  (Dtype)0., output + output_offset_ * g);
#ifdef USE_PROFILE_DISPLAY
		LOG(INFO)<<this->layer_param().name()<<"\t group "<<g<<": "<<timer.MicroSeconds()<<" us (Dense Scheme Timing)";
#endif
#ifdef USE_SNAPSHOT_FEATURE
	  if(num_forward_image_ < 5){
		ostringstream filename_stream;
		//sprintf(filename,"%s.feature%d",this->layer_param().name().c_str(),num_forward_image_);
		filename_stream << this->layer_param().name() << "_group"<<g<<".feature" << num_forward_image_;
		MM_typecode matcode;
		FILE * fp = fopen(filename_stream.str().c_str(), "w+");
		mm_initialize_typecode(&matcode);
		mm_set_matrix(&matcode);
		mm_set_array(&matcode);
		mm_set_real(&matcode);
		mm_set_general(&matcode);

		mm_write_banner(fp, matcode);
		//int M = this->shape(0);//column of the stored matrix
		//int N = this->count()/M;
		mm_write_mtx_array_size(fp, K, N);
		/* NOTE: matrix market files use 1-based indices, i.e. first element
		 of a vector has index 1, not 0.  */
		for (int col=0; col<N; col++) {
			for (int row=0; row<K; row++) {
				fprintf(fp, "%20g\n", (double)(*(col_buff + + col_offset_ * g + row * N + col)) );
			}
		}
		fclose(fp);
	  }
#endif
		break;
	  }
  }
#ifdef USE_SNAPSHOT_FEATURE
	  num_forward_image_ += 1;
#endif

  if (this->layer_param_.convolution_param().conv_mode() == caffe::ConvolutionParameter_ConvMode_DIRECT_SCONV) {
    delete[] input_padded;
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
  if(this->layer_param_.convolution_param().conv_mode() == ConvolutionParameter_ConvMode_LOWERED_CCNMM){
	  Blob<Dtype> input_buf;
	  input_buf.Reshape(1,conv_in_channels_,conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2]);
	  caffe_copy(input_buf.count(), input, input_buf.mutable_cpu_data());
	  conv_im2col_cpu(input_buf.cpu_data(), col_buffer_.mutable_cpu_data(), col_buf_mask_.mutable_cpu_data());
	  col_buff = col_buffer_.gpu_data();
  }else
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  Timer timer;
  timer.Start();
  int col_buf_offset_sum = 0;
  int output_offset_sum = 0;
  int weight_offset_sum = 0;
  for (int g = 0; g < group_; ++g) {
	  switch(this->layer_param_.convolution_param().conv_mode()){
			case caffe::ConvolutionParameter_ConvMode_LOWERED_CSRMM :{
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
				timer.Start();
				caffe_gpu_sparse_csrmm(conv_out_channels_ / group_,
						conv_out_spatial_dim_,
						kernel_dim_,
						(Dtype)1.,
						nz_num_[g],
						nz_weight_values_.gpu_data()+ weight_offset_ * g,
						nz_weight_index_pointers_.gpu_data() + (conv_out_channels_ / group_ + 1) * g,
						nz_weight_indices_.gpu_data()+ weight_offset_ * g,
						col_buff + col_offset_ * g,
						(Dtype)0.,
						output + output_offset_ * g,
						transposed_output_buffer_.mutable_gpu_data());
#ifdef USE_PROFILE_DISPLAY
				LOG(INFO)<<this->layer_param().name()<<"\t group "<<g<<": "<<timer.MicroSeconds()<<" us (Compressed Row Storage Timing)";
#endif
				break;
			}
			case caffe::ConvolutionParameter_ConvMode_LOWERED_CCNMM :{
				  timer.Start();
				  //LOG(INFO)<<"Computing ConvolutionParameter_ConvMode_LOWERED_CCNMM";
				  int left_cols = left_columns_[g];
				  int left_rows = left_rows_[g];
				  timer.Start();
//				  caffe_cpu_cblas_gemm(left_rows, conv_out_spatial_dim_, left_cols,
//						  (Dtype)1., squeezed_weight_buffer_.cpu_data() + weight_offset_sum,
//						  left_cols , col_buff + col_buf_offset_sum,
//						conv_out_spatial_dim_, (Dtype)0., output + output_offset_sum, conv_out_spatial_dim_);
				  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
						  left_rows, conv_out_spatial_dim_, left_cols,
						 (Dtype)1., squeezed_weight_buffer_.gpu_data() + weight_offset_sum,
						 col_buff + col_buf_offset_sum,
						 (Dtype)0., output + output_offset_sum);
#ifdef USE_PROFILE_DISPLAY
				  LOG(INFO)<<this->layer_param().name()<<"\t group "<<g<<": "<<timer.MicroSeconds()<<" us (Concatenation Timing)";
#endif
				  col_buf_offset_sum += left_cols * conv_out_spatial_dim_;
				  output_offset_sum += left_rows * conv_out_spatial_dim_;
				  weight_offset_sum += left_rows*left_cols;
				  //dispatch output feature maps using CPU function
				  if(group_-1 == g){
					  Blob<Dtype> output_buf;
					  output_buf.Reshape(1,1,conv_out_channels_, conv_out_spatial_dim_);
					  caffe_copy(output_buf.count(), output, output_buf.mutable_cpu_data());
					  caffe_cpu_dispatch_rows(conv_out_channels_,conv_out_spatial_dim_,output_buf.mutable_cpu_data(),row_buf_mask_.cpu_data());
					  caffe_copy(output_buf.count(), output_buf.cpu_data(), output);
				  }
				  break;
			}
			default:{
				timer.Start();
				caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
					group_, conv_out_spatial_dim_, kernel_dim_,
						 (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
						 (Dtype)0., output + output_offset_ * g);
#ifdef USE_PROFILE_DISPLAY
				LOG(INFO)<<this->layer_param().name()<<"\t group "<<g<<": "<<timer.MicroSeconds()<<" us (Dense Scheme Timing)";
#endif
				break;
			}
	  }
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
