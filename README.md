# Pedestrian-Detection-on-TX1

##Introduction
We compress the pedestrian detection model from ResNet-200 (63 millions parameter) to our fixed channel ResNet-18 (0.157 million parameter). Our paper utilizes the idea of Knowledge Distillation with extra helps from model confidence and hint layer to achieve 400x compression with 4.9% log-average miss rate drop. For more detail, please refer to our [arXiv paper](https://arxiv.org/abs/1612.00478) and [slides](https://noranart.github.io/files/slides.pptx).

<p align="center">
<img src="https://noranart.github.io/img/composite_kd.png" alt="pipeline" width="500px">
</p>

##Result
Log-average miss rate on Caltech (lower is better)

| Model | Log-avg MR | #Parameters | Time (Titan X) | Memory |
|:-------|:-----:|:-------:|:-------:|:-------:|
| ResNet-200 | 17.5% | 63M | 24ms | 5377MB |
| ResNet-18 | 18.0% | 11M | 3ms | 937MB |
| ResNet-18-Thin | 20.3% | 2.8M | 3ms | 633MB |
| ResNet-18-Small | 22.4% | 0.157M | 3ms | 565MB |
*Results are from the highest improvement method (Hint+Conf).


##Demo
<p align="center">
<a href="https://www.youtube.com/watch?v=36RSc1ZuNvE"><img src="https://img.youtube.com/vi/36RSc1ZuNvE/0.jpg" alt="pipeline" width="500px"></a>
</p>

##Installation
###Training
The networks were trained by [torch-nnet-trainer](https://github.com/jonathanasdf/torch-nnet-trainer/tree/ffd5a933e731556ab9eff7e0d160848166c95a1a). 
Please set up caltech10x according to [Hosang](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/people-detection-pose-estimation-and-tracking/taking-a-deeper-look-at-pedestrians/).

###Testing on TX1
Please build [SquareChnnlFltrs](https://bitbucket.org/rodrigob/doppia) for region proposal and replace
- 2014_eccvw_SquaresChnFtrs_trained_on_Caltech.proto.bin	
- libmonocular_objects_detection.so	

The input is a set of images from video (extract by convert.sh). Use "make forward" for building forward.cpp, and "make run --input($input) --output($output)" to forward the images.


##Citing our model
If you found our model useful, please cite our paper and Knowledge Distillation paper:

    @articles{liu2016ssd,
      title = {In Teacher We Trust: Learning Compressed Models for Pedestrian Detection},
      author = {Shen, J.,{Vesdapunt, N., Boddeti, V.~N. and Kitani, K.~M.},
      journal = {arXiv preprint, arXiv:1612.00478},
      year = {2016}
    }

    @articles{liu2016ssd,
      title = {Distilling the knowledge in a neural network},
      author = {Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
      journal = {arXiv preprint arXiv:1503.02531},
      year = {2015}
    }
