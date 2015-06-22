#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  
  for (int i = 0; i < this->layer_param_.loss_param().class_weight_size(); i++) {
    class_weight_.push_back(this->layer_param_.loss_param().class_weight(i));
  }
  if (class_weight_.size() < 2) {
    LOG(INFO) << "No class_weight specified. Use 1 for both classes." << std::endl;
    class_weight_.clear();
    class_weight_.push_back((Dtype) 1.0);
    class_weight_.push_back((Dtype) 1.0);
  }
  else {
    LOG(INFO) << "positive class weight: " << class_weight_[0] << std::endl
      << "negative class weight: " << class_weight_[1] << std::endl;
  }
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;
  Dtype wc = 0;
  for (int i = 0; i < count; ++i) {
    if (target[i] > 0) {
    // Update the loss only if target[i] is not 0
      loss -= class_weight_[0] * (input_data[i] * ((target[i] > 0) - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0))));
      wc += class_weight_[0];
    }
    else if (target[i] < 0) {
      loss -= class_weight_[1] * (input_data[i] * ((target[i] > 0) - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0))));
      wc += class_weight_[1];
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / wc;
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    Dtype wc = 0;
    for (int i = 0; i < count; ++i) {
      if (target[i] > 0) {
        bottom_diff[i] = class_weight_[0] * (sigmoid_output_data[i] - (target[i] > 0));
        wc += class_weight_[0];
      }
      else if (target[i] < 0) {
        bottom_diff[i] = class_weight_[1] * (sigmoid_output_data[i] - (target[i] > 0));
        wc += class_weight_[1];
      }
      else {
        bottom_diff[i] = 0;
      }
    }
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / wc, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidCrossEntropyLossLayer);
#endif

INSTANTIATE_CLASS(SigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SigmoidCrossEntropyLoss);

}  // namespace caffe
