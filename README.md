# VOF-LSHNet

This is the official code repository for our paper entitled **“VOF-LSHNet”**.

The repository contains the core implementation, the processed feature cache
required to run the model, the semantic prototype file, and the numerical
source data used for the figures. Raw vehicle records, trained checkpoints,
training logs, and experiment orchestration scripts are not included.

## Repository structure

```text
VOF-LSHNet/
|-- core_code/
|   |-- cc_vofrwt.py
|   |-- hierarchical_predictor_v4.py
|   |-- joint_train_v4_6class.py
|   `-- semantic_prototypes_v4.py
|-- source_data/
|   |-- data/
|   |   |-- raw_train.npy
|   |   |-- raw_val.npy
|   |   |-- raw_test.npy
|   |   |-- wave_train.npy
|   |   |-- wave_val.npy
|   |   |-- wave_test.npy
|   |   |-- labels_train.npy
|   |   |-- labels_val.npy
|   |   |-- labels_test.npy
|   |   |-- metadata.json
|   |   `-- semantic_prototypes_v4_data_grounded.npz
|   `-- figure_source_data/
|       `-- variable_order/
|-- README.md
`-- requirements.txt
```

## Environment

The archived experiments used Python 3.8.20 and CUDA 11.8. A CUDA-capable GPU
is recommended for training.

```bash
conda create -n vof-lshnet python=3.8
conda activate vof-lshnet
pip install -r requirements.txt
```

Install a PyTorch build compatible with the CUDA runtime and GPU driver on the
target machine when the pinned build is not appropriate.

The `transformers`, `accelerate`, and `bitsandbytes` packages are required only
when regenerating semantic prototypes. They are not required when using the
released `semantic_prototypes_v4_data_grounded.npz`.

## Six-class prediction task

The model is trained and evaluated on the following six operating-state–fault
classes:

```text
run_normal
charge_normal
run_insulation
charge_insulation
charge_voltage
charge_temperature
```

## Processed data

The files under `source_data/data` are sufficient to train and evaluate the
released model. Each split contains:

- a normalized 64-point raw sequence (`raw_*.npy`);
- a cached variable-order fractional wavelet representation (`wave_*.npy`);
- the corresponding class labels (`labels_*.npy`).

The cache uses five raw signals and a 160-dimensional wavelet representation.
The raw vehicle records used to construct the cache are not distributed
because of data-use and privacy restrictions.

Files such as `paths_val.npy` and `paths_test.npy` are not required by the
training program and must not be included in a public release because source
paths may contain local or identifiable information.

The training arrays are larger than the regular per-file limit of common Git
hosting services. Distribute them using Git LFS, a release asset, or a
research-data repository rather than regular Git objects.

## Run VOF-LSHNet

From the repository root:

```bash
cd core_code
python joint_train_v4_6class.py \
  --cache-dir ../source_data/data \
  --semantic-prototype-path ../source_data/data/semantic_prototypes_v4_data_grounded.npz \
  --output-dir ../outputs/seed_84 \
  --epochs 40 \
  --early-stopping-patience 6 \
  --lr-scheduler-patience 2 \
  --batch-size 64 \
  --num-workers 0 \
  --seed 84 \
  --device cuda
```

On Windows Command Prompt, the same command can be entered on one line:

```bat
python joint_train_v4_6class.py --cache-dir "../source_data/data" --semantic-prototype-path "../source_data/data/semantic_prototypes_v4_data_grounded.npz" --output-dir "../outputs/seed_84" --epochs 40 --early-stopping-patience 6 --lr-scheduler-patience 2 --batch-size 64 --num-workers 0 --seed 84 --device cuda
```

To reproduce the multi-seed evaluation, repeat the command with seeds `84`,
`512`, and `2026`, using a different output directory for each run.

Training writes the following files to `--output-dir`:

```text
best_predictor_v4_6class.pt
result_v4_6class.json
training_curves_v4_6class.csv
semantic_prototypes_v4_data_grounded.npz
```

### CPU smoke test

The following command checks that the cache, semantic prototypes, and training
pipeline can be loaded. It is not intended to reproduce the reported results.

```bash
python joint_train_v4_6class.py \
  --cache-dir ../source_data/data \
  --semantic-prototype-path ../source_data/data/semantic_prototypes_v4_data_grounded.npz \
  --output-dir ../outputs/cpu_smoke_test \
  --epochs 1 \
  --batch-size 8 \
  --num-workers 0 \
  --device cpu
```

## Semantic prototypes

The released semantic prototype file can be used directly. To regenerate it,
obtain a compatible Hugging Face causal language model separately and run:

```bash
python semantic_prototypes_v4.py \
  --model-path /path/to/language-model \
  --output ../source_data/data/semantic_prototypes_v4_data_grounded.npz \
  --batch-size 1 \
  --max-length 384 \
  --last-layers 4
```

The default generation mode uses 4-bit model loading. Add `--no-4bit` when
4-bit loading is unavailable. Users are responsible for obtaining the language
model and complying with its license.

## Figure source data

The numerical data used for the main and supplementary figures are provided in:

```text
source_data/figure_source_data/
```

These files contain plotting values rather than rendered figure images.
Record-level source data must be checked for local paths, vehicle identifiers,
and other restricted information before release.

## Reproducibility

The same random seed, processed cache, semantic prototype file, software
versions, and training arguments should be used when comparing runs. GPU and
CUDA operations can introduce small numerical differences, so independently
trained models are expected to reproduce the reported multi-seed statistics
rather than identical checkpoints or identical values at every epoch.

## Data availability

The processed data and figure source data distributed with this repository are
provided to support reproduction of the reported analyses. The original
vehicle records are not publicly distributed here. Refer to the paper’s Data
Availability statement for access conditions.
