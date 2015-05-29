#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


namespace caffe {


template <typename Dtype>
void ExpandLayer<Dtype>::Forward_gpu (const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (this->layer_param_.use_cpu()) {
    Forward_cpu(bottom, top);
    return;
  }
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  int spatial_dim = expand_w_ * expand_h_;
  for (int n = 0; n < bottom[0]->num(); n++) {
    for (int c = 0; c < channels_; c++) {
      caffe_gpu_set(spatial_dim, *bottom_data, 
            top_data);
      bottom_data += bottom[0]->offset(0,1);
      top_data += top[0]->offset(0,1);
    }
  }
}

template <typename Dtype>
__global__ void ExpandBackward(const int nthreads, const Dtype* top_diff,
    const int channels, const int height, const int width, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = index % channels;
    int n = index / channels;
    Dtype sumval = 0;
    top_diff += (n * channels + c) * height * width;
    for (int i = 0; i < height*width; ++i) {
      sumval += top_diff[i];
    }
    bottom_diff[index] = sumval;
  } 
}

template <typename Dtype>
void ExpandLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (this->layer_param_.use_cpu()) {
    Backward_cpu(top, propagate_down, bottom);
    return;
  }
  if (!propagate_down[0])
    return;
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int nthreads = top[0]->num()*channels_;
  ExpandBackward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, top_diff, channels_, expand_h_, expand_w_, bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(ExpandLayer);

} // end of namespace caffe
