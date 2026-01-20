# PHGD: Preference-guided Heterogeneous Graph Denoising for Robust Recommendation

## Requirements
- python==3.9.20
- pytorch==2.1.0
- dgl==2.4.0
- cuda==118

### Others
```python
pip install -r requirement.txt
```

## Running on Yelp, DoubanBook, Yelp, and DoubanMovie Datasets
```python
python main_PHGD.py --lr 0.001 --dataset LastFM --gpu 0 --num_workers 12 --batch 1024 --cl_rate 0.09 --IB_rate 0.0002 --sigma 0.3 --n_layers 1 --han_layers 1
python main_PHGD.py --lr 0.0005 --dataset Amazon --gpu 0 --num_workers 12 --batch 1024 --cl_rate 0.09 --IB_rate 0.0002 --sigma 0.25 --n_layers 1 --han_layers 1
python main_PHGD.py --lr 0.001 --dataset Movielens --gpu 0 --num_workers 12 --batch 1024 --cl_rate 0.2 --IB_rate 0.0002 --sigma 0.25 --n_layers 1 --han_layers 1
python main_PHGD.py --lr 0.001 --dataset Yelp --gpu 0 --num_workers 12 --batch 1024 --cl_rate 0.1 --IB_rate 5.0 --sigma 0.4 --n_layers 2 --han_layers 1
python main_PHGD.py --lr 0.001 --dataset DoubanBook --gpu 0 --num_workers 12 --batch 5180 --cl_rate 0.1 --IB_rate 0.02 --sigma 0.2 --n_layers 1 --han_layers 2
```
