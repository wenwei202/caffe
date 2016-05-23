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
#include "caffe/layers/conv_relu_pool_layer.hpp"
#include "caffe/util/conv.hpp"
#include "CSR.hpp"
#include "reordering/BFSBipartite.hpp"

namespace caffe {

template <typename Dtype>
BaseConvolutionLayer<Dtype>::BaseConvolutionLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      /*weight_interleaved_(NULL), */input_padded_(NULL), output_scratch_(NULL)
      /*, input_scratch_(NULL), output_colmajor_scratch_(NULL)*/ {
  //is_sparse_format_weights_ = false;
  is_concatenating_weights_features_ = false;
}

template <typename Dtype>
BaseConvolutionLayer<Dtype>::~BaseConvolutionLayer()
{
//  free(weight_interleaved_);
  free(input_padded_);
  free(output_scratch_);
//  free(input_scratch_);
//  free(output_colmajor_scratch_);

  for (int i = 0; i < weight_rowptr_blocked_.size(); ++i) {
    free(weight_rowptr_blocked_[i]);
    free(weight_colidx_blocked_[i]);
    free(weight_values_blocked_[i]);
  }

  for (int i = 0; i < weight_rowptr_.size(); ++i) {
    free(weight_rowptr_[i]);
    free(weight_colidx_[i]);
    free(weight_values_[i]);
  }

//  for (int i = 0; i < weight_blockptr_colmajor_.size(); ++i) {
//    free(weight_blockptr_colmajor_[i]);
//    free(weight_kidx_colmajor_[i]);
//    free(weight_values_colmajor_[i]);
//  }

//  for (int i = 0; i < weight_colidx_interleaved_.size(); ++i) {
//    free(weight_rowptr_interleaved_[i]);
//    free(weight_colidx_interleaved_[i]);
//    free(weight_values_interleaved_[i]);
//  }
}

bool barrier_initialized = false;

template <>
void BaseConvolutionLayer<double>::WeightAlign(){
  NOT_IMPLEMENTED;
}

template <>
void BaseConvolutionLayer<float>::WeightAlign(){

  if (!barrier_initialized) {
    int nthreads = omp_get_max_threads();
    int nthread_groups = nthreads;
#ifdef __AVX512F__
    nthread_groups = NTILES;
#else
  //  nthread_groups = nthreads/2;
#endif

    assert(nthreads%nthread_groups == 0);
    int nthreads_per_group = nthreads/nthread_groups;
    if (nthread_groups != nthreads) {
      for (int i = 0; i < nthread_groups; ++i) {
        barriers[i] = new synk::Barrier(1, nthreads_per_group);
      }
#pragma omp parallel
      {
        assert(omp_get_num_threads() == nthreads);

        int tid = omp_get_thread_num();
        int gid = tid/nthreads_per_group;
        int tid_in_group = tid%nthreads_per_group;

        barriers[gid]->init(tid_in_group);
      }
    }
    barrier_initialized = true;
  }

	CHECK_EQ(this->blobs_[0]->num_axes(),4);//caffe now supports any dimension
	//is_sparse_format_weights_ = false;
	const LayerParameter& layerparam = this->layer_param();
	LOG(INFO)<<"layer\t"<<layerparam.name()<<"\t"<<"has sparsity of "<< this->blobs_[0]->GetSparsity();
//	this->blobs_[0]->WriteToNistMMIO(layerparam.name()+".weight");

	ConvolutionParameter conv_param = this->layer_param_.convolution_param();
	const int M = this->blobs_[0]->shape(0)/group_;
	const int N = this->blobs_[0]->count(1,4);
	const int weight_offset = this->blobs_[0]->count()/group_;
	const int row_offset = this->blobs_[0]->shape(0)/group_ + 1;
	int masked_col_num = 0;
	int left_cols = 0;
	float group_sparsity = 0;
	switch(conv_param.conv_mode()){
		case caffe::ConvolutionParameter_ConvMode_LOWERED_CSRMM:
			LOG(INFO)<<"ConvolutionParameter_ConvMode_LOWERED_CSRMM";

      weight_rowptr_.resize(group_);
      weight_colidx_.resize(group_);
      weight_values_.resize(group_);

			for (int g = 0; g < group_; ++g) {
        int nnz = 0;
        for (int i = 0; i < M*N; ++i) {
          if (this->blobs_[0]->cpu_data()[i] != 0) ++nnz;
        }

        posix_memalign((void **)&weight_rowptr_[g], 4096, sizeof(int)*(M + 1));
        posix_memalign((void **)&weight_colidx_[g], 4096, sizeof(int)*nnz);
        posix_memalign((void **)&weight_values_[g], 4096, sizeof(float)*nnz);

				caffe_cpu_sparse_dense2csr(M, N,
						this->blobs_[0]->mutable_cpu_data() + weight_offset * g,
						weight_values_[g],
						weight_colidx_[g],
						weight_rowptr_[g]);
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
			group_sparsity = (float)masked_col_num/(float)col_buf_mask_.count();
			LOG(INFO) << Layer<float>::layer_param().name() << " column sparsity: " << group_sparsity;
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
        int kernel_h = kernel_shape_.cpu_data()[0];
        int kernel_w = kernel_shape_.cpu_data()[1];

        int col_block_size = COL_BLOCK;
        if (conv_in_channels_ == 1024) {
          // Overfeat conv5
          col_block_size = 128;
        }
        int ncolblocks = conv_in_channels_/col_block_size;
        LOG(INFO) << "ncolblocks " << ncolblocks;
        weight_rowptr_blocked_.resize(ncolblocks);
        weight_colidx_blocked_.resize(ncolblocks);
        weight_values_blocked_.resize(ncolblocks);
        std::vector<int> nnzs_of_col_blocks(ncolblocks, 0);

        weight_rowptr_.resize(group_);
        weight_colidx_.resize(group_);
        weight_values_.resize(group_);

//        weight_blockptr_colmajor_.resize(group_);
//        weight_kidx_colmajor_.resize(group_);
//        weight_values_colmajor_.resize(group_);

        weight_rowptr_split_.resize(group_);
        weight_colidx_split_.resize(group_);
        weight_values_split_.resize(group_);

//        weight_rowptr_interleaved_.resize(ncolblocks);
//        weight_colidx_interleaved_.resize(ncolblocks);
//        weight_values_interleaved_.resize(ncolblocks);

				for (int g = 0; g < group_; ++g) {
				  int nnz = 0;
				  for (int i = 0; i < M*N; ++i) {
				    if (this->blobs_[0]->cpu_data()[weight_offset*g + i] != 0) ++nnz;
				  }

	        posix_memalign((void **)&weight_rowptr_[g], 4096, sizeof(int)*(M + 1));
	        posix_memalign((void **)&weight_colidx_[g], 4096, sizeof(int)*nnz);
	        posix_memalign((void **)&weight_values_[g], 4096, sizeof(float)*nnz);

					// first create a CSR matrix as for LOWERED_CSRMM
					caffe_cpu_sparse_dense2csr(M, N,
							this->blobs_[0]->mutable_cpu_data() + weight_offset * g,
	            weight_values_[g],
	            weight_colidx_[g],
	            weight_rowptr_[g]);

					// declare variables for sparsity statistics
					vector<vector<int> > nnz_per_channel_pair(M);
					for(int i = 0; i < M; ++i) {
						nnz_per_channel_pair[i] = vector<int>(conv_in_channels_, 0);
					}
          vector<int> nnz_per_oc_fiber(N, 0);
          assert(N == conv_in_channels_/group_*kernel_h*kernel_w);
					int num_of_non_zero_kernels = 0;
          int num_of_non_zero_out_channels = 0;

					const int *rowptr = weight_rowptr_[g];
          assert(nnz == rowptr[M]);
//          posix_memalign((void **)&weight_blockptr_colmajor_[g], 4096, sizeof(int)*(conv_in_channels_/group_*M + 1));
//          memset(weight_blockptr_colmajor_[g], 0, sizeof(int)*(conv_in_channels_/group_*M + 1));
//          posix_memalign((void **)&weight_kidx_colmajor_[g], 4096, sizeof(int)*nnz);
//          posix_memalign((void **)&weight_values_colmajor_[g], 4096, sizeof(float)*nnz);
          int VLEN = 16;

          posix_memalign((void **)&weight_rowptr_split_[g], 4096, sizeof(int)*(ncolblocks*M*kernel_w + 1));
          memset(weight_rowptr_split_[g], 0, sizeof(int)*(ncolblocks*M*kernel_w + 1));
          posix_memalign((void **)&weight_colidx_split_[g], 4096, sizeof(int)*nnz);
          posix_memalign((void **)&weight_values_split_[g], 4096, sizeof(float)*nnz);

//          posix_memalign((void **)&weight_rowptr_interleaved[g], 4096, sizeof(int)*(M + 1));
//          memcpy(weight_rowptr_interleaved[g], rowptr, sizeof(int)*(M + 1));
//          posix_memalign((void **)&weight_colidx_interleaved_[g], 4096, sizeof(int)*nnz);
//          posix_memalign((void **)&weight_values_interleaved[g], 4096, sizeof(float)*nnz);
//          memcpy(weight_values_interleaved[g], rowptr, sizeof(int)*(M + 1));

          int col_major_ic_block = get_col_major_ic_block(nnz, M, conv_in_channels_/group_);
          assert(conv_in_channels_/group_%col_major_ic_block == 0);
          LOG(INFO) << "col_major_ic_block = " << col_major_ic_block;

					// transform the indices for direct convolution
					int *colidx = weight_colidx_[g];
					for (int oc = 0; oc < M; ++oc) {
						for (int j = rowptr[oc]; j < rowptr[oc + 1]; ++j) {
							int col = colidx[j];

							int kernel_col = col%kernel_w;
							int kernel_row = (col/kernel_w)%kernel_h;
							int ic = col/(kernel_w*kernel_h);
							assert(ic < conv_in_channels_/group_);

							colidx[j] = (ic*(height + pad_h) + kernel_row)*(width + pad_w) + kernel_col;

							int bcol = ic/col_block_size + ncolblocks/group_*g;
							++nnzs_of_col_blocks[bcol];

							++weight_rowptr_split_[g][(ic/col_block_size*M + oc)*kernel_w + (kernel_w - 1 - kernel_col) + 1];

//							int bcol_colmajor = ic/col_major_ic_block;
//							++weight_blockptr_colmajor_[g][bcol_colmajor*M + oc + 1];

							++nnz_per_channel_pair[oc][ic];
              ++nnz_per_oc_fiber[col];
						}
            if (rowptr[oc + 1] > rowptr[oc]) {
              num_of_non_zero_out_channels++;
            }

						for (int in_channel = 0; in_channel < conv_in_channels_; ++in_channel) {
							if (nnz_per_channel_pair[oc][in_channel] != 0) {
								++num_of_non_zero_kernels;
							}
						}
					}

//					for (int i = 1; i < conv_in_channels_/group_/col_major_ic_block*M; ++i) {
//					  weight_blockptr_colmajor_[g][i + 1] += weight_blockptr_colmajor_[g][i];
//					}
//					assert(weight_blockptr_colmajor_[g][conv_in_channels_/group_/col_major_ic_block*M] == nnz);
					for (int i = 1; i < ncolblocks*M*kernel_w; ++i) {
					  weight_rowptr_split_[g][i + 1] += weight_rowptr_split_[g][i];
					}
					assert(weight_rowptr_split_[g][ncolblocks*M*kernel_w] == nnz);

//          for (int out_channel = 0; out_channel < M; ++out_channel) {
//            int nnz_of_oc = 0;
//            for (int i = 0; i < conv_in_channels_/group_/col_major_ic_block; ++i) {
//              nnz_of_oc += weight_blockptr_colmajor_[g][i*M + out_channel + 1] - weight_blockptr_colmajor_[g][i*M + out_channel];
//            }
//            assert(nnz_of_oc == rowptr[out_channel + 1] - rowptr[out_channel]);
//          }

          int num_of_non_zero_oc_fibers = 0;
          for (int i = 0 ; i < N; ++i) {
            if (nnz_per_oc_fiber[i] > 0) ++num_of_non_zero_oc_fibers;
          }

          std::vector<int> kernel_non_zero_hist(kernel_w*kernel_h, 0);
          for (int in_channel = 0; in_channel < conv_in_channels_/group_; ++in_channel) {
            int cnt = 0;
            for (int i = in_channel*kernel_w*kernel_h; i < (in_channel + 1)*kernel_w*kernel_h; ++i) {
              kernel_non_zero_hist[i - in_channel*kernel_w*kernel_h] += nnz_per_oc_fiber[i];
            }
          }

          std::stringstream stream;
          stream << "kernel_non_zero_hist = ";
          for (int i = 0; i < kernel_w*kernel_h; ++i) {
            stream << i << ":" << kernel_non_zero_hist[i] << " ";
          }
          LOG(INFO) << stream.str();

					LOG(INFO) << "oc-mode fiber sparsity " << 1 - (double)num_of_non_zero_oc_fibers/N;
					LOG(INFO) << "oc-mode slice sparsity " << 1 - (double)num_of_non_zero_out_channels/M;
					LOG(INFO) << "k-mode fiber sparsity " << 1 - (double)num_of_non_zero_kernels/(M*(conv_in_channels_/group_));
					LOG(INFO) << "nnz = " << nnz;

					SpMP::CSR A(M, conv_in_channels_/group_, num_of_non_zero_kernels);
					nnz = 0;
					A.rowptr[0] = 0;
					for (int out_channel = 0; out_channel < M; ++out_channel) {
					  for (int in_channel = 0; in_channel < conv_in_channels_/group_; ++in_channel) {
					    if (nnz_per_channel_pair[out_channel][in_channel] != 0) {
					      A.colidx[nnz] = in_channel;
					      ++nnz;
					    }
					  }
					  A.rowptr[out_channel + 1] = nnz;
					}

					SpMP::CSR *AT = A.transpose();
					int *rowPerm = new int[M], *rowInversePerm = new int[M];
					int *colPerm = new int[conv_in_channels_/group_], *colInversePerm = new int[conv_in_channels_/group_];
					bfsBipartite(A, *AT, rowPerm, rowInversePerm, colPerm, colInversePerm);
					FREE(A.diagptr);
					SpMP::CSR *AReordered = A.permute(colPerm, rowInversePerm);
					SpMP::CSR *ATReordered = AReordered->transpose();

					LOG(INFO) << "conv_in_channels_ = " << conv_in_channels_ << " Average width of oc x ic matrix = " << A.getAverageWidth() << " " << AT->getAverageWidth();
					LOG(INFO) << "Average width after reordering = " << AReordered->getAverageWidth() << " " << ATReordered->getAverageWidth();

					delete[] rowPerm;
					delete[] rowInversePerm;
					delete[] colPerm;
					delete[] colInversePerm;
					delete AT;
					delete AReordered;
					delete ATReordered;
				} // for each group

				for (int i = 0; i < ncolblocks; ++i) {
				  posix_memalign((void **)&weight_rowptr_blocked_[i], 4096, sizeof(int)*(M + 1));
				  posix_memalign((void **)&weight_colidx_blocked_[i], 4096, sizeof(int)*nnzs_of_col_blocks[i]);
				  posix_memalign((void **)&weight_values_blocked_[i], 4096, sizeof(float)*nnzs_of_col_blocks[i]);
//				  posix_memalign((void **)&weight_colidx_interleaved_[i], 4096, sizeof(int)*nnzs_of_col_blocks[i]);
				  nnzs_of_col_blocks[i] = 0;
				  weight_rowptr_blocked_[i][0] = 0;
				}

        int stride_h = stride_.cpu_data()[0];
        int stride_w = stride_.cpu_data()[1];
        int dilation_h = dilation_.cpu_data()[0];
        int dilation_w = dilation_.cpu_data()[1];

        const int output_h = (height + 2 * pad_h -
            (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
        const int output_w = (width + 2 * pad_w -
            (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

        const int SCRATCH_SIZE_PER_IC = output_h*16;

        int max_col_major_ic_block = 0;
				for (int g = 0; g < group_; ++g) {
          const int *rowptr = weight_rowptr_[g];
          int *colidx = weight_colidx_[g];
          float *values = weight_values_[g];

          int nnz = rowptr[M];
          int col_major_ic_block = get_col_major_ic_block(nnz, M, conv_in_channels_/group_);
          max_col_major_ic_block = std::max(max_col_major_ic_block, col_major_ic_block);

				  for (int oc = 0; oc < M; ++oc) {
				    for (int j = rowptr[oc]; j < rowptr[oc + 1]; ++j) {
              int c = colidx[j];

              int kernel_col = c%(width + pad_w);
              int kernel_row = c/(width + pad_w)%(height + pad_h);
              int ic = c/(width + pad_w)/(height + pad_h);
              int bcol = ic/col_block_size + ncolblocks/group_*g;

              weight_colidx_blocked_[bcol][nnzs_of_col_blocks[bcol]] = c;
              weight_values_blocked_[bcol][nnzs_of_col_blocks[bcol]] = values[j];
              const int VLEN = 16;
//              weight_colidx_interleaved_[bcol][nnzs_of_col_blocks[bcol]] = c*VLEN;
              nnzs_of_col_blocks[bcol]++;

              int splitid = (ic/col_block_size*M + oc)*kernel_w + (kernel_w - 1 - kernel_col);
              weight_colidx_split_[g][weight_rowptr_split_[g][splitid]] =
                  (ic*(height + pad_h) + kernel_row)*16;
              weight_values_split_[g][weight_rowptr_split_[g][splitid]] = values[j];
              ++weight_rowptr_split_[g][splitid];

//              int blockid = ic/col_major_ic_block*M + oc;
//              int offset = weight_blockptr_colmajor_[g][blockid];
//              weight_kidx_colmajor_[g][offset] =
//                  ((ic%col_major_ic_block*kernel_w + kernel_col)*(height + pad_h) + kernel_row)*16;
//              weight_values_colmajor_[g][offset] = values[j];
//              ++weight_blockptr_colmajor_[g][blockid];
				    }

				    for (int i = ncolblocks/group_*g; i < ncolblocks/group_*(g + 1); ++i) {
				      weight_rowptr_blocked_[i][oc + 1] = nnzs_of_col_blocks[i];
				    }
				  }

//          for (int i = conv_in_channels_/group_/col_major_ic_block*M - 1; i > 0; --i) {
//            weight_blockptr_colmajor_[g][i] = weight_blockptr_colmajor_[g][i - 1];
//          }
//          weight_blockptr_colmajor_[g][0] = 0;
          for (int i = ncolblocks*M*kernel_w - 1; i > 0; --i) {
            weight_rowptr_split_[g][i] = weight_rowptr_split_[g][i - 1];
          }
          weight_rowptr_split_[g][0] = 0;
//          for (int out_channel = 0; out_channel < M; ++out_channel) {
//            int nnz_of_oc = 0;
//            for (int i = 0; i < conv_in_channels_/group_/col_major_ic_block; ++i) {
//              nnz_of_oc += weight_blockptr_colmajor_[g][i*M + out_channel + 1] - weight_blockptr_colmajor_[g][i*M + out_channel];
//            }
//            assert(nnz_of_oc == rowptr[out_channel + 1] - rowptr[out_channel]);
//          }
				} // for each group



	      posix_memalign((void **)&output_scratch_, 4096, sizeof(float)*OC_BLOCK*width*16*omp_get_max_threads());

//	      posix_memalign(
//	          (void **)&input_scratch_,
//	          4096,
//	          sizeof(float)*omp_get_max_threads()*max_col_major_ic_block*kernel_h*kernel_w*SCRATCH_SIZE_PER_IC);
//	      memset(input_scratch_, 0, sizeof(float)*omp_get_max_threads()*max_col_major_ic_block*kernel_h*kernel_w*SCRATCH_SIZE_PER_IC);

//#ifdef COL_MAJOR_OC_BLOCK
//	      posix_memalign((void **)&output_colmajor_scratch_, 4096, sizeof(float)*omp_get_max_threads()*COL_MAJOR_OC_BLOCK*SCRATCH_SIZE_PER_IC);
//#else
//	      posix_memalign((void **)&output_colmajor_scratch_, 4096, sizeof(float)*omp_get_max_threads()*M*SCRATCH_SIZE_PER_IC);
//#endif

//	      posix_memalign((void **)&input_interleaved_, 4096, sizeof(float)*this->num_*conv_in_channels_*(height + pad_h)*(width + pad_w));
//	      memset(input_interleaved_, 0, sizeof(float)*this->num_*conv_in_channels_*(height + pad_h)*(width + pad_w));
//	      posix_memalign((void **)&output_interleaved_, 4096, sizeof(float)*this->num_*conv_out_channels_*output_h*output_w);

#ifdef __AVX512F__
	      if (height == 13 || height == 12) {
          int input_aligned_len = conv_in_channels_ * (height + pad_h)*16 + pad_h*16;
          posix_memalign((void **)&input_aligned_, 4096, sizeof(float)*omp_get_max_threads()*input_aligned_len);
          memset(input_aligned_, 0, sizeof(float)*omp_get_max_threads()*input_aligned_len);
	      }
//	      else
#endif
	      {
	        int input_padded_len = conv_in_channels_ * (height + pad_h) * (width + pad_w) + pad_h * (width + 2 * pad_w);
	        posix_memalign((void **)&input_padded_, 4096, sizeof(float)*omp_get_max_threads()*input_padded_len);
	        memset(input_padded_, 4096, sizeof(float)*omp_get_max_threads()*input_padded_len);
	      }

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
//            sizeof(float) * M * conv_in_channels_ * kernel_size_aligned);
//        for (int g = 0; g < group_; ++g) {
//          for (int out_channel = 0; out_channel < M; ++out_channel) {
//            for (int in_channel = 0; in_channel < conv_in_channels_/group_; ++in_channel) {
//              memcpy(
//                  weight_aligned_ + ((g*M + out_channel)*(conv_in_channels_/group_) + in_channel)*kernel_size_aligned,
//                  this->blobs_[0]->cpu_data() + weight_offset*g + (out_channel*(conv_in_channels_/group_) + in_channel)*kernel_h*kernel_w,
//                  sizeof(float)*kernel_h*kernel_w);
//            }
//          }
//        }
//
//        int kernel_w_aligned = (kernel_w + 7)/8*8;
//        int kernel_size_aligned2 = (kernel_h*kernel_w_aligned + 15)/16*16;
//        posix_memalign(
//            (void **)&weight_aligned2_,
//            1024,
//            sizeof(float) * M * conv_in_channels_ * kernel_size_aligned2);
//        for (int g = 0; g < group_; ++g) {
//          for (int out_channel = 0; out_channel < M; ++out_channel) {
//            for (int in_channel = 0; in_channel < conv_in_channels_/group_; ++in_channel) {
//              for (int h = 0; h < kernel_h; ++h) {
//                memcpy(
//                    weight_aligned2_ + ((g*M + out_channel)*(conv_in_channels_/group_) + in_channel)*kernel_size_aligned2 + h*kernel_w_aligned,
//                    this->blobs_[0]->cpu_data() + weight_offset*g + ((out_channel*(conv_in_channels_/group_) + in_channel)*kernel_h + h)*kernel_w,
//                    sizeof(float)*kernel_w);
//                memset(
//                    weight_aligned2_ + ((g*M + out_channel)*(conv_in_channels_/group_) + in_channel)*kernel_size_aligned2 + h*kernel_w_aligned + kernel_w,
//                    0,
//                    sizeof(float)*(kernel_w_aligned - kernel_w));
//              }
//            }
//          }
//        }

        assert(M%8 == 0);
        posix_memalign(
            (void **)&weight_interleaved_,
            4096,
            sizeof(float) * M * conv_in_channels_ * kernel_h * kernel_w);
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
		this->blobs_[0]->Disconnect(Blob<float>::ELTWISE);
	}else if(layerparam.connectivity_mode() == caffe::LayerParameter_ConnectivityMode_DISCONNECTED_GRPWISE){
		LOG(INFO)<<"weights lying in all-zero groups of "<<layerparam.name()<<" are frozen";
		this->blobs_[0]->Disconnect(Blob<float>::GRPWISE, group_);
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
//	  nz_weight_values_.Reshape(1, 1, 1, this->blobs_[0]->count());//nonzero elements
//	  nz_weight_indices_.Reshape(1,1,1,nz_weight_values_.count());//index of nonzero
//	  nz_weight_index_pointers_.Reshape(1,1,1,this->blobs_[0]->shape(0)+group_);//pointer(index) of indices
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

template<>
void BaseConvolutionLayer<float>::forward_cpu_gemm(const float* input,
    const float* weights, float* output, int batch_idx, bool skip_im2col) {
  const float* col_buff = input;

  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();

  int nthread_groups = nthreads;
#ifdef __AVX512F__
  if (this->layer_param_.convolution_param().conv_mode() != caffe::ConvolutionParameter_ConvMode_DIRECT_DCONV) {
//    if (height != 27) nthread_groups = NTILES;
  }
#endif
  assert(nthreads%nthread_groups == 0);
  int nthreads_per_group = nthreads/nthread_groups;
  int gid = tid/nthreads_per_group;
  int tid_in_group = tid%nthreads_per_group;

  float *input_padded, *input_aligned;
  int input_padded_len;
  if (this->layer_param_.convolution_param().conv_mode() == caffe::ConvolutionParameter_ConvMode_DIRECT_SCONV) {
    // JSP: pad boundaries with zeros to avoid checking boundary conditions

	  int height = conv_input_shape_.cpu_data()[1];
	  int width = conv_input_shape_.cpu_data()[2];
	  int pad_h = pad_.cpu_data()[0];
	  int pad_w = pad_.cpu_data()[1];

	  input_padded_len = conv_in_channels_ * (height + pad_h) * (width + pad_w) + pad_h * (width + 2 * pad_w);
	  int input_aligned_len = conv_in_channels_ * (height + pad_h) * 16 + pad_h * 16;
	  if (pad_h == 0 && pad_w == 0) {
	    input_padded = (float *)input;
	  }
	  else {
	    if (tid == 0) padding_time -= omp_get_wtime();

	    input_padded = input_padded_ + input_padded_len*gid;
	    input_aligned = input_aligned_ + input_aligned_len*gid;

	    int c_per_thread = (conv_in_channels_ + nthreads_per_group - 1)/nthreads_per_group;
	    int cbegin = std::min(c_per_thread*tid_in_group, conv_in_channels_);
	    int cend = std::min(cbegin + c_per_thread, conv_in_channels_);

#ifdef __AVX512F__
      if (height == 13 || height == 12) {
        for (int in_channel = cbegin; in_channel < cend; ++in_channel) {
          for (int input_row = 0; input_row < height; ++input_row) {
            memcpy(
                input_aligned + (in_channel * (height + pad_h) + input_row + pad_h) * 16 + pad_w,
                input + (in_channel * height + input_row) * width,
                sizeof(float) * width);
          }
        }
      }
//      else
#endif
      {
        for (int in_channel = cbegin; in_channel < cend; ++in_channel) {
          for (int input_row = 0; input_row < height; ++input_row) {
            memcpy(
                input_padded + (in_channel * (height + pad_h) + input_row + pad_h) * (width + pad_w) + pad_w,
                input + (in_channel * height + input_row) * width,
                sizeof(float) * width);
          }
        }
      }

      if (nthread_groups != nthreads) barriers[gid]->wait(tid_in_group);

      if (tid == 0) padding_time += omp_get_wtime();
	  }
  }
  else if (this->layer_param_.convolution_param().conv_mode() == caffe::ConvolutionParameter_ConvMode_DIRECT_DCONV) {
    // JSP: currently, don't need to do anything
  }
  else if (!is_1x1_ ||  is_concatenating_weights_features_) {
    if (tid == 0) im2col_time -= omp_get_wtime();

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

    if (tid == 0) im2col_time += omp_get_wtime();
  }

  int offset_sum = 0;
  for (int g = 0; g < group_; ++g) {
	  const int M = conv_out_channels_ /group_;
	  const int N = conv_out_spatial_dim_;
	  const int K = kernel_dim_;
	  const int row_offset = conv_out_channels_ /group_ + 1;
	  int left_cols = 0;
	  switch(this->layer_param_.convolution_param().conv_mode()){
	  case caffe::ConvolutionParameter_ConvMode_LOWERED_CSRMM :
		  caffe_cpu_sparse_mmcsr(M,
				  N,
				  K,
				  (float)1.,
				  weight_values_[g],
				  weight_colidx_[g],
				  weight_rowptr_[g],
				  weight_rowptr_[g] + 1,
				  col_buff + col_offset_ * g,
				  (float)0.,output + output_offset_ * g);
		  break;
	  case caffe::ConvolutionParameter_ConvMode_LOWERED_CCNMM :
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

      const int weight_offset = this->blobs_[0]->count()/group_;

      if (height == 227 && width == 227 && pad_h == 0 && pad_w == 0 && stride_h == 4 && stride_w == 4 && kernel_w == 11 && kernel_h == 11 && dilation_h == 1 && dilation_w == 1 && conv_in_channels_/group_ == 3) {
        //      const int weight_offset = this->blobs_[0]->count()/group_;
        //      const float *weight = this->blobs_[0]->cpu_data() + weight_offset * g;
        //      int kernel_size_aligned = (kernel_h * kernel_w + 15)/16*16;
        //      const float *weight = weight_aligned_ + g*M*(conv_in_channels_/group_)*kernel_size_aligned;
        //      int kernel_w_aligned = (kernel_w + 7)/8*8;
        //      int kernel_size_aligned2 = (kernel_w_aligned*kernel_h + 15)/16*16;
        //      const float *weight = weight_aligned2_ + g*M*(conv_in_channels_/group_)*kernel_size_aligned2;

        const float *weight = weight_interleaved_;
        assert(std::string(type()) == "ConvolutionReLUPoolLRN");
        caffe_cpu_dconv<float>(
            input + conv_in_channels_/group_ * g * height * width,
            conv_in_channels_, height, width,
            pad_h, pad_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            weight_interleaved_ + weight_offset * g,
            kernel_h, kernel_w,
            this->blobs_[1]->cpu_data(), bias_multiplier_.cpu_data(),
            ((ConvolutionReLUPoolLRNLayer<float> *)this)->pool_top_[0]->mutable_cpu_data() + ((ConvolutionReLUPoolLRNLayer<float> *)this)->pool_top_[0]->offset(0, 1)*(conv_out_channels_*batch_idx + M*g),
            ((ConvolutionReLUPoolLRNLayer<float> *)this)->max_idx_.mutable_cpu_data() + ((ConvolutionReLUPoolLRNLayer<float> *)this)->pool_top_[0]->offset(0, 1)*(conv_out_channels_*batch_idx + M*g),
            output + output_offset_ * g,
            M);
      }
      else {
        caffe_cpu_dconv<float>(
            input + conv_in_channels_/group_ * g * height * width,
            conv_in_channels_/group_, height, width,
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

      const int output_h = (height + 2 * pad_h -
          (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
      const int output_w = (width + 2 * pad_w -
          (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

		  const int *rowptr = weight_rowptr_[g];
		  const float *values = weight_values_[g];
		  const int *colidx = weight_colidx_[g];

		  int ncolblock = weight_rowptr_blocked_.size()/group_;
      int nnz = rowptr[M];
      int col_major_ic_block = get_col_major_ic_block(nnz, M, conv_in_channels_/group_);

		  if (height == 27 && width == 27 && pad_h == 2 && pad_w == 2 && stride_h == 1 && stride_w == 1 && kernel_w == 5 && kernel_h == 5 && dilation_h == 1 && dilation_w == 1) {
		    // 2nd layer of AlexNet fused with bias term and pooling

		    assert(std::string(type()) == "ConvolutionReLUPoolLRN");
        caffe_cpu_sconv<float>(
            input_padded + conv_in_channels_/group_ * g * (height + pad_h) * (width + pad_w),
            input + conv_in_channels_/group_ * g * height * width,
            conv_in_channels_/group_,
            height, width,
            pad_h, pad_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            rowptr, colidx, values,
            kernel_h, kernel_w,
            (const int **)(&weight_rowptr_blocked_[0] + g*ncolblock),
            (const int **)(&weight_colidx_blocked_[0] + g*ncolblock),
            (const float **)(&weight_values_blocked_[0] + g*ncolblock),
            ncolblock,
//            NULL, NULL, NULL,
//            input_scratch_, output_colmajor_scratch_,
            this->blobs_[1]->cpu_data(), bias_multiplier_.cpu_data(),
            ((ConvolutionReLUPoolLRNLayer<float> *)this)->pool_top_[0]->mutable_cpu_data() + ((ConvolutionReLUPoolLRNLayer<float> *)this)->pool_top_[0]->offset(0, 1)*(conv_out_channels_*batch_idx + M*g),
            ((ConvolutionReLUPoolLRNLayer<float> *)this)->max_idx_.mutable_cpu_data() + ((ConvolutionReLUPoolLRNLayer<float> *)this)->pool_top_[0]->offset(0, 1)*(conv_out_channels_*batch_idx + M*g),
            output + output_offset_ * g,
            M,
            output_scratch_ + tid*OC_BLOCK*width*16,
            col_major_ic_block,
            input_aligned + conv_in_channels_/group_ * g * (height + pad_h) * 16,
            weight_rowptr_split_[g], weight_colidx_split_[g], weight_values_split_[g]);
		  }
		  else if (height == 28 && width == 28 && pad_h == 0 && pad_w == 0 && stride_h == 1 && stride_w == 1 && kernel_w == 5 && kernel_h == 5 && dilation_h == 1 && dilation_w == 1) {
		    // 2nd layer of OverFeat fused with bias term and pooling
		    assert(std::string(type()) == "ConvolutionReLUPool");

		    const float *bias = this->blobs_[1]->cpu_data();
		    const float *bm = bias_multiplier_.cpu_data();
		    float *pool_top = (*((ConvolutionReLUPoolLayer<float> *)this)->top_ptr_)[0]->mutable_cpu_data() + (*((ConvolutionReLUPoolLayer<float> *)this)->top_ptr_)[0]->offset(0, 1)*(conv_out_channels_*batch_idx + M*g);
		    int *mask = ((ConvolutionReLUPoolLayer<float> *)this)->max_idx_.mutable_cpu_data() + (*((ConvolutionReLUPoolLayer<float> *)this)->top_ptr_)[0]->offset(0, 1)*(conv_out_channels_*batch_idx + M*g);

        caffe_cpu_sconv<float>(
            input_padded + conv_in_channels_/group_ * g * (height + pad_h) * (width + pad_w),
            input + conv_in_channels_/group_ * g * height * width,
            conv_in_channels_/group_,
            height, width,
            pad_h, pad_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            rowptr, colidx, values,
            kernel_h, kernel_w,
            (const int **)(&weight_rowptr_blocked_[0] + g*ncolblock),
            (const int **)(&weight_colidx_blocked_[0] + g*ncolblock),
            (const float **)(&weight_values_blocked_[0] + g*ncolblock),
            ncolblock,
//            NULL, NULL, NULL,
//            input_scratch_, output_colmajor_scratch_,
            bias, bm,
            pool_top,
            mask,
            output + output_offset_ * g,
            M,
            output_scratch_ + tid*OC_BLOCK*width*16,
            col_major_ic_block,
            input_aligned + conv_in_channels_/group_ * g * (height + pad_h) * 16,
            weight_rowptr_split_[g], weight_colidx_split_[g], weight_values_split_[g]);
		  }
		  else {
#ifdef VECTORIZE_OVER_INPUTS
        const int VLEN = 16;
		    if (std::string(type()) == "ConvolutionReLU")
		    {
		      if (batch_idx%VLEN == 0) {
            assert(output_offset_ == conv_out_channels_/group_*output_h*output_w);
            sconv345_vectorize_over_inputs(
                input_interleaved_ + (batch_idx/VLEN*conv_in_channels_ + conv_in_channels_/group_*g)*(height + pad_h)*(width + pad_w)*VLEN,
                (const int **)(&weight_rowptr_blocked_[0] + g*ncolblock),
                (const int **)(&weight_colidx_interleaved_[0] + g*ncolblock),
                (const float **)(&weight_values_blocked_[0] + g*ncolblock),
                ncolblock,
                this->blobs_[1]->cpu_data(),
                output_interleaved_ + (batch_idx/VLEN*conv_out_channels_ + conv_out_channels_/group_*g)*output_h*output_w*VLEN,
                M, conv_in_channels_/group_);
		      }
		    }
		    else
#endif
		    {
          caffe_cpu_sconv<float>(
              input_padded + conv_in_channels_/group_ * g * (height + pad_h) * (width + pad_w),
              input + conv_in_channels_/group_ * g * height * width,
              conv_in_channels_/group_,
              height, width,
              pad_h, pad_w,
              stride_h, stride_w,
              dilation_h, dilation_w,
              rowptr, colidx, values,
              kernel_h, kernel_w,
              (const int **)(&weight_rowptr_blocked_[0] + g*ncolblock),
              (const int **)(&weight_colidx_blocked_[0] + g*ncolblock),
              (const float **)(&weight_values_blocked_[0] + g*ncolblock),
              ncolblock,
//              &weight_blockptr_colmajor_[g][0], &weight_kidx_colmajor_[g][0], &weight_values_colmajor_[g][0],
//              input_scratch_, output_colmajor_scratch_,
              this->blobs_[1]->cpu_data(), bias_multiplier_.cpu_data(),
              NULL, NULL,
              output + output_offset_ * g,
              M,
              output_scratch_ + tid*OC_BLOCK*width*16,
              col_major_ic_block,
              input_aligned + conv_in_channels_/group_ * g * (height + pad_h) * 16,
              weight_rowptr_split_[g], weight_colidx_split_[g], weight_values_split_[g]);
		    }
		  }

		  break;
	  }
	  default:
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
