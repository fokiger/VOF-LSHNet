# VOF-LSHNet

The repository provides the core implementation of VOF-LSHNet

## Environment requirement

We recommend using a Conda environment.

```bash
conda create -n vof-lshnet python=3.10
conda activate vof-lshnet
```

Install PyTorch according to your CUDA version, and then install the remaining
packages:

```bash
pip install numpy pandas scikit-learn tqdm
pip install transformers accelerate
```

The semantic-prototype generation code uses 4-bit language-model loading by
default. Install `bitsandbytes` if this mode is required:

```bash
pip install bitsandbytes
```


## Repository structure

```text
VOF-LSHNet
|-- core_code
|   |-- cc_vofrwt.py
|   |-- hierarchical_predictor_v4.py
|   |-- joint_train_v4_6class.py
|   |-- semantic_prototypes_v4.py
|-- source_data
|-- README.md
```

## Dataset

### Raw data

The vehicle dataset used in this study will be processed, anonymized, and reorganized for public release in the near future. The processed datasets required to reproduce the reported results are provided in the form of `.npy` files, including the input features, labels, and associated metadata used for model training and evaluation. 

### Feature cache

The cache directory contains files such as:

```text
v4_feature_cache
|-- raw_train.npy
|-- raw_val.npy
|-- raw_test.npy
|-- wave_train.npy
|-- wave_val.npy
|-- wave_test.npy
|-- labels_train.npy
|-- labels_val.npy
|-- labels_test.npy
|-- metadata.json
```

## Generate semantic prototypes

Replace `/path/to/language-model` with the model directory on your computer and
run:

```bash
cd "core _code"
python semantic_prototypes_v4.py \
  --model-path /path/to/language-model \
  --output ../semantic_prototypes_v4_data_grounded.npz \
  --batch-size 1 \
  --max-length 384 \
  --last-layers 4
```

## Run VOF-LSHNet

### Train

After preparing the feature cache and semantic prototypes, run:

```bash
cd "core _code"
python joint_train_v4_6class.py \
  --cache-dir ../v4_feature_cache \
  --semantic-prototype-path ../semantic_prototypes_v4_data_grounded.npz \
  --output-dir ../outputs/seed_42 \
  --epochs 40 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --weight-decay 1e-5 \
  --seed 84 \
  --device cuda
```

The training outputs, including the best checkpoint, training curves, run
configuration, and test metrics, are saved in the directory specified by
`--output-dir`.





