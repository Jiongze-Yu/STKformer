# Real-Time Smoke Detection with Split Top-k Transformer and Adaptive Dark Channel Prior in Foggy Environments

## 🔥 🔥 

## Datasets
```
datasets
├── DSDF
│   ├── train
│   └── val
├── CIS
│   ├── train
│   ├── test
│   └── val
├── UIW
│   ├── train
│   ├── test
│   └── val
└── CDSD
    ├── classification
    │   ├── train
    │   └── val
    └── detection
        ├── train
        └── val
```

## Training
```
python -m torch.distributed.launch --nproc_per_node 2 --master_port 123456 main.py --batch_size 64 --model_name STKformer_0_75_100_25 --data_path ./datasets/DSDF --tag STKformer_0_75_100_25_100epochs_2gpu_64 --load_ckpt ./weights/model-best.pth
```

## Testing
```
python evaluate.py --batch_size 1 --model_name STKformer_0_75_100_25 --data_path ./datasets/DSDF --load_ckpt ./weights/STKformer_0_75_100_25/model-best.pth
python evaluate.py --batch_size 1 --model_name STKformer_0_75_100_25 --data_path ./datasets/DSDF --load_ckpt ./weights/STKformer_0_75_100_25/model-best.trt
```

## Pre-trained Models
