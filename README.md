# ECG: Edge-aware Point Cloud Completion with Graph Convolution

Pytorch Implementation
Evaluated with python 3.5 and pytorch 1.2

![prediction example](https://github.com/paul007pl/ECG/blob/master/misc/qualitative.png)


### Introduction
This work is based on our paper "ECG: Edge-aware Point Cloud Completion with Graph Convolution"

Scanned 3D point clouds for real-world scenes often suffer from noise and incompletion. Observing that prior point cloud shape completion networks overlook local geometric features, we propose our ECG - an Edge-aware point cloud Completion network with Graph convolution, which facilitates fine-grained 3D point cloud shape generation with multi-scale edge features. Our ECG consists of two consecutive stages: 1)skeleton generation and 2) details refinement. Each stage is a generation sub-network conditioned on the input incomplete point cloud. The first stage generates coarse skeletons to facilitate capturing useful edge features against noisy measurements.  Subsequently, we design a deep hierarchical encoder with graph convolution to propagate multi-scale edge features for local geometric details refinement. To preserve local geometrical details while upsampling, we propose the Edge-aware Feature Expansion (EFE) module to smoothly expand/upsample point features by emphasizing their local edges. Extensive experiments show that our ECG significantly outperforms previous state-ofthe-art (SOTA) methods for point cloud completion.


### Installation
1. Install required python libs
2. Downloading corresponding dataset (e.g. ShapeNet dataset; TopNet dataset; or Cascade dataset)
3. Compile pytorch 3rd libs

### Citation
If you find our work useful in your research, please cite:

	@ARTICLE{9093117,
	author={L. {Pan}},
	journal={IEEE Robotics and Automation Letters}, 
	title={ECG: Edge-aware Point Cloud Completion with Graph Convolution}, 
	year={2020},
	volume={5},
	number={3},
	pages={4392-4398},}

