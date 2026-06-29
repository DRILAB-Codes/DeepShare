# PointNet++ AutoEncoder Baseline

This module trains a PointNet++-style autoencoder:

```text
one robot partial LiDAR scan -> full obstacle boundary point cloud
```

## Train

```bash
pip install torch numpy pyyaml matplotlib
python -m ae.train_ae --config ae/configs/ae_config.yaml
```

## Compare SSG and MSG

Copy `ae/configs/ae_config.yaml`, then change:

```yaml
model:
  encoder_mode: msg
training:
  out_dir: out/ae_msg
```

## Visualize

```bash
python -m ae.visualize_recon \
  --checkpoint out/ae_ssg/model_best.pt \
  --data_dir data/ae/val \
  --save_path out/ae_ssg/recon.png
```

## Expected data format

Each scene JSON should contain:

```text
sample["robots"][i]["pose"]
sample["robots"][i]["lidar"]
sample["obstacle"]["polygon"]
```

Each robot scan becomes one training sample.
