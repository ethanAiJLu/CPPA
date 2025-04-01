# Cross-Modal Prompt-Guided Source-Free Domain Adaptation for Remote Sensing Scene Recognition

This repository provides the official implementation of our paper "Cross-Modal Prompt-Guided Source-Free Domain Adaptation for Remote Sensing Scene Recognition" which introduces a novel approach for domain adaptation in remote sensing imagery using vision-language models.

## Overview

CPPA (Cross-Modal Prompt-guided Prototype Alignment) is a multi-modal prompt learning framework that enables effective knowledge transfer between domains for remote sensing scene recognition without requiring target domain labels. Our method leverages both visual and textual modalities to create more robust and adaptable models that can generalize across different remote sensing datasets.

Key features:

- Deep cross-modal interactions between visual and textual prompts
- Source-free domain adaptation for remote sensing scene recognition
- Novel prototype alignment mechanism for cross-domain knowledge transfer
- State-of-the-art performance on remote sensing benchmarks
- Compatible with CLIP and other vision-language models

## Implementation Details

In our implementation:

- We adopted the pre-trained CLIP model with the ViT-B/16 image encoder as our backbone network
- Training parameters:
  - Initial learning rate: 0.0035
  - Learning rate scheduler: Cosine annealing
  - Training epochs on target domain: 25
  - Prompt depth (L): 9
  - Number of context tokens (M): 8
- Hardware: Two NVIDIA RTX 3060 GPUs (12 GB each)
- Framework: PyTorch with Python 3.8

## Installation and Environment Setup

This codebase is tested on Ubuntu 20.04.2 LTS with Python 3.8 and PyTorch framework. Our experiments were conducted on a setup with two NVIDIA RTX 3060 GPUs, each with 12 GB of memory. Follow these steps to create the environment and install dependencies:

### Environment Setup

```bash
# Recommended Python version: 3.8
# Tested on Ubuntu 20.04.2 LTS and Windows 10/11

# Create a conda environment
conda create -y -n cppa python=3.8

# Activate the environment
conda activate cppa

# Install PyTorch (requires version >= 1.8.1) and torchvision
# For different CUDA versions, refer to https://pytorch.org/
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

### Install Dassl Library

```bash
# Clone the repository
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install the library (no need to rebuild if the source code is modified)
python setup.py develop
cd ..
```

For more details about Dassl Library，you can see this [repository](https://github.com/KaiyangZhou/Dassl.pytorch).

### Install CPPA

```bash
# Clone the CPPA repository
git clone https://github.com/yourusername/CPPA.git
cd CPPA/

# Install requirements
pip install -r requirements.txt
```

## Remote Sensing Datasets

Our experiments were conducted on several widely-used remote sensing scene classification datasets:

### UCM (UC Merced Land-Use)

- **Description**: A collection of 21 land-use classes with 100 images per class
- **Image Size**: 256×256 pixels
- **Resolution**: 0.3m per pixel
- **Source**: Manually extracted from USGS National Map Urban Area Imagery
- **Classes**: Include agricultural, airplane, baseball diamond, beach, buildings, etc.
- **Download**: http://weegee.vision.ucmerced.edu/datasets/landuse.html

### AID (Aerial Image Dataset)

- **Description**: A large-scale dataset with 30 scene categories
- **Image Size**: 600×600 pixels
- **Images per Class**: Between 200 to 400 images
- **Source**: Google Earth imagery
- **Total Images**: ~10,000 images from different regions around the world
- **Download**: https://captain-whu.github.io/AID/

### NWPU-RESISC45

- **Description**: A benchmark for remote sensing image scene classification
- **Image Size**: 256×256 pixels
- **Classes**: 45 scene categories with 700 images per class
- **Total Images**: 31,500 images
- **Source**: Google Earth, with varied spatial resolution (0.2m to 30m per pixel)
- **Download**: OneDrive link available at https://www.tensorflow.org/datasets/catalog/resisc45

### WHU-RS19

- **Description**: A relatively small dataset with 19 different scene classes
- **Image Size**: 600×600 pixels
- **Images per Class**: 50 samples per class
- **Total Images**: 950 images
- **Source**: Google Earth imagery with high spatial resolution
- **Download**: https://captain-whu.github.io/BED4RS/

### RSSCN7

- **Description**: Remote sensing scene classification dataset with 7 categories
- **Image Size**: 400×400 pixels
- **Images per Class**: 400 samples per class
- **Total Images**: 2,800 images
- **Classes**: Grassland, farmland, industrial regions, river/lake, forest, residential, and parking lot
- **Source**: Google Earth satellite imagery
- **Download**: Available on Kaggle at https://www.kaggle.com/datasets/yangpeng1995/rsscn7

These datasets represent a diverse range of remote sensing scenes with varying resolution, scale, and geographic distribution, making them ideal for evaluating the generalization capability of our cross-domain adaptation approach.

## Data Preparation

To use our framework with the remote sensing datasets, you'll need to organize them in the following experiment settings structure:

```
Copy$DATA/
|–– setting1/
|   |–– ucm/
|   |–– whu/
|
|–– setting2_1/
|   |–– aid/
|   |–– ucm/
|
|–– setting2_2/
|   |–– nwpu/
|   |–– ucm/
|
|–– setting2_3/
|   |–– aid/
|   |–– nwpu/
|
|–– setting3/
    |–– aid/
    |–– rsscn7/
    |–– ucm/
    |–– whu/
```

Each dataset should have class subfolders with the relevant images:

```
Copydataset_name/
|–– class1/
|   |–– image1.jpg
|   |–– image2.jpg
|   |–– ...
|–– class2/
|   |–– image1.jpg
|   |–– image2.jpg
|   |–– ...
|–– ...
```

For domain adaptation experiments, we used various combinations of these datasets:

- Setting 1: Bidirectional adaptation between UCM and WHU
  - UCM → WHU
  - WHU → UCM
- Setting 2.1: Bidirectional adaptation between AID and UCM
  - AID → UCM
  - UCM → AID
- Setting 2.2: Bidirectional adaptation between NWPU and UCM
  - NWPU → UCM
  - UCM → NWPU
- Setting 2.3: Bidirectional adaptation between AID and NWPU
  - AID → NWPU
  - NWPU → AID
- Setting 3: Multi-domain adaptation across UCM, AID, WHU, and RSSCN7 (12 different transfer directions)
  - UCM → AID
  - UCM → WHU
  - UCM → RSSCN7
  - AID → UCM
  - AID → WHU
  - AID → RSSCN7
  - WHU → UCM
  - WHU → AID
  - WHU → RSSCN7
  - RSSCN7 → UCM
  - RSSCN7 → AID
  - RSSCN7 → WHU

If you have your own datasets, you can organize them according to the dataset storage format mentioned above.

## Training and Evaluation

### Domain Adaptation Experiments

We provide bash scripts in the `scripts/cppa/` directory for training and evaluating our models on different domain adaptation tasks. Here are some examples:

```bash
# Setting 1: 
# WHU → UCM , UCM → WHU
bash ./scripts/cppa/setting_1.sh

# Setting 2: 
# AID → UCM , UCM → AID
# NWPU → UCM , UCM → NWPU
# AID → NWPU , NWPU → AID
bash ./scripts/cppa/setting_2_call.sh

# setting 3: 
# UCM → AID , UCM → WHU , UCM → RSSCN7 
# AID → UCM , AID → WHU , AID → RSSCN7
# WHU → UCM , WHU → AID , WHU → RSSCN7
# RSSCN7 → UCM , RSSCN7 → AID , RSSCN7 → WHU
bash ./scripts/cppa/setting_3.sh
```

### Configuration

You can customize the training process by modifying the configuration files in scripts:

```bash
export CUDA_VISIBLE_DEVICES=0

# Please change './data' to your path to data.
python train.py --root ./data --seed 1 --trainer CPPA \
    --config-file "$config_file" \
    --dataset-config-file "$dataset_config_file" \
    --output-dir "$output_dir" \
    --source-domains "$source_domains" \
    --target-domains "$target_domains" \
    --opts TRAINER.CPPA.N_CTX 16 \
    TRAINER.CPPA.PROMPT_DEPTH 9 \
    TRAINER.CPPA.FUSING mean \
    TRAINER.CPPA.PS True \
    DATASET.SUBSAMPLE_CLASSES all \
    DATASET.NUM_SHOTS 8 \
    TRAINER.CPPA.CTX_INIT ""
```

You can also change the model config in `configs\trainers\CPPA\vit_b16.yaml`

```yaml
# exsample
DATALOADER:
  TRAIN_X:
    BATCH_SIZE: 4
  TEST:
    BATCH_SIZE: 16
  NUM_WORKERS: 1

TRAINER:
  CPPA:
    N_CTX: 16
    CTX_INIT: "a photo of a"
    PREC: "fp16"
    PROMPT_DEPTH: 9
    FUSING: "mean"
    PS: True

```

### Results Analysis

After training completes, use our parsing script to calculate average performance:

```bash
# Calculate average accuracy across seeds
python parse_test_res.py output/ucm_aid
```

The script will display:

- Individual run accuracies
- Mean accuracy
- Standard deviation or confidence interval
- Summary statistics

## Citation

If you find our work useful, please consider citing:

```
@inproceedings{author2025cppa,
  title={Cross-Modal Prompt-Guided Source-Free Domain Adaptation for Remote Sensing Scene Recognition},
  author={Li, Yongguang and Yin, Ziqi},
  booktitle={Proceedings of the Conference},
  year={2025}
}
```

## Acknowledgements

Our implementation builds upon [Co-CoOp and CoOp ](https://github.com/KaiyangZhou/CoOp), [MaPLe ](https://github.com/muzairkhattak/multimodal-prompt-learning), and [CMPA ](https://github.com/GingL/CMPA/tree/main). We thank the authors for releasing their code.

This project is released under the MIT License.