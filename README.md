# Semantics Disentangling for Generalized Zero-shot Learning 

This is the official implementation for paper 
> Zhi Chen, Yadan Luo, Ruihong Qiu, Zi Huang, Jingjing Li, Zheng Zhang.  
**Semantics Disentangling for Generalized Zero-shot Learning**  
_International Conference on Computer Vision (ICCV) 2021_.

[Semantics Disentangling for Generalized Zero-shot Learning](https://arxiv.org/pdf/2101.07978.pdf)

![](architecture.png)

Abstract: Generalized zero-shot learning (GZSL) aims to classify samples under the assumption that some classes are not 
observable during training. To bridge the gap between the seen and unseen classes, most GZSL methods attempt to associate 
the visual features of seen classes with attributes or to generate unseen samples directly. Nevertheless, the visual 
features used in the prior approaches do not necessarily encode semantically related information that the shared 
attributes refer to, which degrades the model generalization to unseen classes. To address this issue, in this paper, 
we propose a novel semantics disentangling framework for the generalized zero-shot learning task (SDGZSL), where the 
visual features of unseen classes are firstly estimated by a conditional VAE and then factorized into semantic-consistent 
and semantic-unrelated latent vectors. In particular, a total correlation penalty is applied to guarantee the independence 
between the two factorized representations, and the semantic consistency of which is measured by the derived relation 
network. Extensive experiments conducted on four GZSL benchmark datasets have evidenced that the semantic-consistent 
features disentangled by the proposed SDGZSL are more generalizable in tasks of canonical and generalized zero-shot 
learning. 


## Requirements
The implementation runs on

- Python 3.6

- torch 1.3.1

- Numpy

- Sklearn

- Scipy

## Usage

Put your [datasets](https://drive.google.com/file/d/1KxFC6T_kGKCNx1JyX2FOaSimA0DOcU_I/view?usp=sharing) in SDGZSL_data folder and run the scripts:

The extracted features for APY and AWA datasets are from [[1]](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/feature-generating-networks-for-zero-shot-learning), 
FLO and CUB datasets are from [[2]](https://github.com/yunlongyu/EPGN). For the fine-tuned features, AWA,FLO and CUB are from [[3]](https://github.com/akshitac8/tfvaegan). 
The APY fine-tuned features are extracted from us.

[1] Xian, Yongqin, et al. "Feature generating networks for zero-shot learning." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

[2] Yu, Yunlong, et al. "Episode-based prototype generating network for zero-shot learning." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

[3] Narayan, Sanath, et al. "Latent embedding feedback and discriminative features for zero-shot classification." ECCV 2020.

## Citation:
If you find this useful, please cite our work as follows:
```
@inproceedings{chen2021semantics,
	title={Semantics Disentangling for Generalized Zero-shot Learning},
	author={Chen, Zhi and Luo, Yadan and Qiu, Ruihong and Huang, Zi and Li, Jingjing and Zhang, Zheng},
	booktitle={ICCV},
	year={2021}
}
```