# DynaMixer: A Vision MLP Architecture with Dynamic Mixing[[arxiv]](https://arxiv.org/pdf/2201.12083.pdf)
This is a Pytorch implementation of our paper DynaMixer, [ICML 2022](https://icml.cc/).

## Comparison with Recent MLP-like Models
|     Model     | Sub-model | Parameters | Top 1 Acc. |
|:-------------:|:---------:|:----------:|:----------:|
|   Cycle-MLP	|     T	    |     28M	 |    81.3    |
|      ViP	    |  Small/7	|     25M	 |    81.5    |
|   Hire-MLP    |     S     |     33M    |    82.1    |
|   DynaMixer   |     S     |     26M    |    82.7    |
|   Cycle-MLP	|     S	    |     50M	 |    82.9    |
|      ViP	    |  Medium/7	|     55M	 |    82.7    |
|   Hire-MLP    |     B     |     58M    |    83.2    |
|   DynaMixer   |     M     |     57M    |    83.7    |
|   Cycle-MLP	|     B	    |     88M	 |    83.4    |
|      ViP	    |  Large/7	|     88M	 |    83.2    |
|   Hire-MLP    |     L     |     96M    |    83.8    |
|   DynaMixer   |     L     |     97M    |    84.3    |

## Requirements
### Environment
```
torch==1.9.0
torchvision>=0.10.0
pyyaml
timm==0.4.12
fvcore
apex if you use 'apex amp'
```
### Data
data prepare: ImageNet with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```
## Training
Command line for training on 8 GPUs (V100)

train dynamixer_s:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 /path/to/imagenet --model dynamixer_s -b 256 -j 8 --opt adamw --epochs 300 --sched cosine --apex-amp --img-size 224 --drop-path 0.1 --lr 2e-3 --weight-decay 0.05 --remode pixel --reprob 0.25 --aa rand-m9-mstd0.5-inc1 --smoothing 0.1 --mixup 0.8 --cutmix 1.0 --warmup-lr 1e-6 --warmup-epochs 20
```
train dynamixer_m:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 /path/to/imagenet --model dynamixer_m -b 128 -j 8 --opt adamw --epochs 300 --sched cosine --apex-amp --img-size 224 --drop-path 0.1 --lr 2e-3 --weight-decay 0.05 --remode pixel --reprob 0.25 --aa rand-m9-mstd0.5-inc1 --smoothing 0.1 --mixup 0.8 --cutmix 1.0 --warmup-lr 1e-6 --warmup-epochs 20
```
train dynamixer_l:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 /path/to/imagenet --model dynamixer_l -b 64 -j 8 --opt adamw --epochs 300 --sched cosine --apex-amp --img-size 224 --drop-path 0.3 --lr 2e-3 --weight-decay 0.05 --remode pixel --reprob 0.25 --aa rand-m9-mstd0.5-inc1 --smoothing 0.1 --mixup 0.8 --cutmix 1.0 --warmup-lr 1e-6 --warmup-epochs 20
```
## Reference
You may want to cite:
```
@article{DBLP:journals/corr/abs-2201-12083,
  author    = {Ziyu Wang and
               Wenhao Jiang and
               Yiming Zhu and
               Li Yuan and
               Yibing Song and
               Wei Liu},
  title     = {DynaMixer: {A} Vision {MLP} Architecture with Dynamic Mixing},
  journal   = {CoRR},
  volume    = {abs/2201.12083},
  year      = {2022},
  url       = {https://arxiv.org/abs/2201.12083},
  eprinttype = {arXiv},
  eprint    = {2201.12083},
  timestamp = {Wed, 02 Feb 2022 15:00:01 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2201-12083.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
## Acknowledgement
The code is based on following repos:  
https://github.com/Andrew-Qibin/VisionPermutator  
https://github.com/ShoufaChen/CycleMLP.  
Thanks for their wonderful works.

## License
Dynamixer is released under MIT License.
