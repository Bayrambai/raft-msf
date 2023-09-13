# Self-Supervised Monocular Scene Flow Estimation

This repository is the official PyTorch implementation of the paper:  

&nbsp;&nbsp;&nbsp;[**RAFT-MSF: Self-Supervised Monocular Scene Flow Using Recurrent Optimizer**](https://link.springer.com/article/10.1007/s11263-023-01828-4)  
&nbsp;&nbsp;&nbsp; Bayram Bayramli [Junhwa Hur](https://hurjunhwa.github.io) and Lu Hongtao

- Contact: bayram.xiaolong[at]gmail.com  

## Installation and Dataset
For installation and configuring dataset please refer to [Self-Supervised Monocular Scene Flow Estimation](https://github.com/visinf/self-mono-sf)

## Training and Inference

**For training**, you can simply run the following script file:

`./train_monosf_selfsup_kitti_raw.sh`

**For evaluation the pretrained model (RAFT-MSF)**, you can simply run the following script file:

`./eval_monosf_selfsup_kitti_train.sh`

## Pretrained Models 

The **[ckpt](ckpt/)** folder contains the checkpoints of the pretrained model.  

## Acknowledgement

Please cite our paper if you use our source code.  

```bibtex
@article{Bayramli2022RAFTMSFSM,
  title={RAFT-MSF: Self-Supervised Monocular Scene Flow using Recurrent Optimizer},
  author={Bayram Bayramli and Junhwa Hur and Hongtao Lu},
  journal={International Journal of Computer Vision},
  year={2023},
  url={https://doi.org/10.1007/s11263-023-01828-4}
}
```
The overall code framework adapted from [Self-MONO-SF](https://github.com/visinf/self-mono-sf) and [RAFT](https://github.com/princeton-vl/RAFT). We thank the authors for their contribution.
