# LibTorch C++ ResNet CIFAR Example

## Introduction

ResNet implementation, training, and inference using LibTorch C++ API. Because there is no native implementation even for the simplest data augmentation and learning rate scheduler, the ResNet18 model accuracy on CIFAR10 dataset is only around 74% whereas the same ResNet18 model could achieve ~87% accuracy on the same dataset with some simple data augmentation and learning rate scheduler (87% accuracy is still low because the first 7x7 convolutional layer used in the original ResNet was optimized for ImageNet dataset rather than CIFAR10 dataset). The ResNet18 inference latency using LibTorch C++ API is ~3.0 ms per image, which is slightly faster than the inference latency ~3.5 ms per image using PyTorch Python API. However, it is still way too slow. In practice, we use highly optimized inference engines, such as TensorRT. The saved model from LibTorch C++ API cannot be used for PyTorch Python API and vice versa. LibTorch C++ API is not as rich as PyTorch Python API and its implementation really takes way too much time. The performance benefits that LibTorch C++ API brought is almost negligible over PyTorch Python API.

Taken together, it is not recommended to use LibTorch C++ API unless there are some special use cases.

## Usages

### Run Docker Container

```
$ docker pull nvcr.io/nvidia/pytorch:21.03-py3
$ docker run -it --rm --gpus all --ipc=host -v $(pwd):/mnt nvcr.io/nvidia/pytorch:21.03-py3
```

### Download Dataset

```
$ cd /mnt/
$ mkdir -p dataset
$ cd dataset/
$ bash download-cifar10-binary.sh
```

### Build Application

```
$ cmake -B build -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
$ cmake --build build --config Release
```

### Run Application

```
$ cd build/src/
$ ./resnet-cifar-demo
```
