# Graph Convolutional Matrix Completion

Implementation of Graph Convolutional Matrix Completion

Rianne van den Berg, Thomas N. Kipf, Max Welling, [Graph Convolutional Matrix Completion](https://arxiv.org/abs/1706.02263) (2017)

## Installation

```bash
python setup.py install
```

## Requirements

  * Python 2.7
  * TensorFlow (1.4)
  * pandas


## Usage

To reproduce the experiments mentioned in the paper you can run the following commands:

**Movielens 100K on official split with features**
```bash
python train.py -d ml_100k --accum stack -do 0.7 -nleft -nb 2 -e 1000 --features --feat_hidden 10 --testing
```

**Movielens 100K on official split without features**
```bash
python train.py -d ml_100k --accum stack -do 0.7 -nleft -nb 2 -e 1000 --testing
```

## Cite

Please cite our paper if you use this code in your own work:

```
@article{vdberg2017graph,
  title={Graph Convolutional Matrix Completion},
  author={van den Berg, Rianne and Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1706.02263},
  year={2017}
}
```
