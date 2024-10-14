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

<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>DSDF</th>
    <th>CIS</th>
    <th>UIW</th>
    <th>CDSD</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Baidu Cloud</td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
    <td> <a href="">Download </a> </td>
  </tr>
</tbody>
</table>

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
<table>
<thead>
  <tr>
    <th>Weights</th>
    <th>ckpt</th>
    <th>tensorrt</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Baidu Cloud</td>
    <td> <a href="https://pan.baidu.com/s/1728RkFAG8tYlpF5OkoGc0A?pwd=b32l ">Download </a> </td>
    <td> <a href="https://pan.baidu.com/s/1kuYaKuQSgCHmTxgcrthM7w?pwd=1nqm ">Download </a> </td>
  </tr>
</tbody>
</table>
