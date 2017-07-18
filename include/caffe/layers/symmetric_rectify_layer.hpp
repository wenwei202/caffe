#ifndef CAFFE_SYMMETRIC_RECTIFY_LAYER_HPP_
#define CAFFE_SYMMETRIC_RECTIFY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Parameterized Symmetric Rectifying non-linearity @f$
 *        y_i = x_i - threshold; if x_i > threshold
 *            = x_i + threshold; if x_i < -threshold
 *            = 0 otherwise.
 *        @f$. Its properties are 1) threshold are
 *        learnable though backprop and 2) threshold can vary across
 *        channels.
 */
template <typename Dtype>
class SymmetricRectifyLayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides SymmetricRectifyParameter symmetric_rectify_param,
   *     with SymmetricRectifyLayer options:
   *   - filler (\b optional, FillerParameter,
   *     default {'type': constant 'value':0.0001}).
   *   - channel_shared (\b optional, default false).
   *     threholds are shared across channels.
   */
  explicit SymmetricRectifyLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SymmetricRectify"; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the SymmetricRectify inputs.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool channel_shared_;
  Blob<Dtype> multiplier_;  // dot multiplier for backward computation of params
  Blob<Dtype> backward_buff_;  // temporary buffer for backward computation
  //Blob<Dtype> bottom_memory_;  // memory for in-place computation
};

}  // namespace caffe

#endif  // CAFFE_SYMMETRIC_RECTIFY_LAYER_HPP_
