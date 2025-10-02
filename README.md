# XCalib2  

**Official implementation of the paper:** *Unsupervised Cameras Calibration for Thermal-Visible Image Registration*  

[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-lightgrey.svg)](LICENSE)  

---

## Table of Contents

- [Introduction](#introduction)  
- [Highlights](#highlights)  
- [Repository Structure](#repository-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Data Preparation](#data-preparation)  
  - [Running Experiments](#running-experiments)  
  - [Evaluation & Visualization](#evaluation--visualization)  
- [Results & Examples](#results--examples)  
- [Configuration & Options](#configuration--options)  
- [Citation](#citation)  
- [License](#license)  
- [Acknowledgements](#acknowledgements)  

---

## Introduction

XCalib2 provides a full implementation of our **unsupervised calibration method** for aligning thermal and visible camera views.  
Unlike traditional approaches, this method enables **camera calibration without ground-truth correspondences**, making it suitable for real-world scenarios where manual calibration is impractical.  

This repository was developed to accompany the paper:  
> *Unsupervised Cameras Calibration for Thermal-Visible Image Registration*  
> (Authors, Venue, Year)  

You can use this code to reproduce our results, compare with baseline methods, or adapt the technique to your own thermal-visible calibration problems.

---

## Highlights

- ✅ **Unsupervised calibration pipeline** — no manual correspondence labeling 
- 📊 **Quantitative and qualitative evaluation** on multiple datasets
- ⚙️ **Modular design** — separate components for cameras optimization and registration 
- 🎥 **Pipeline scripts** for both training and testing included  
- 📂 **Visualization support** for qualitative results  
- 🔧 **YAML-based configuration** for flexible experimentation  

---

## Repository Structure

```text
.
├── backbone/           # Neural network backbones or feature extractors  
├── examples/           # Example input data and usage scripts  
├── flow/               # Flow estimation / matching modules  
├── frame_sampler/      # Frame sampling strategies  
├── loss/               # Loss functions (alignment, consistency, etc.)  
├── misc/               # Utility modules (IO, transforms, etc.)  
├── model/              # Calibration / registration model definitions  
├── options/            # Configuration and argument parsing  
├── Run.py              # Main entry point for training / inference  
├── test.py             # Script for evaluating model performance  
├── environment.yml     # Conda environment specification  
└── LICENSE             # License file (Unlicense)  
````

---

## Installation

We recommend using a Conda environment.

```bash
git clone https://github.com/Azorgz/XCalib2.git
cd XCalib2

# Create conda env
conda env create -f environment.yml
conda activate XCalib2

# (Alternatively) pip install dependencies
pip install -r requirements.txt
```
---
## Pretrained Weights

Pretrained weights are required to run inference or reproduce results.
Use the provided shell script to automatically download them:

```bash 
sh download_weights.sh
```


This will download the weights into the appropriate folder (usually checkpoints/ or similar).
Make sure the script has execution permissions; if not, run:

```bash 
chmod +x download_weights.sh
```

---

## Usage

### Data Preparation & Configuration

1. Collect thermal-visible image pairs (or datasets) you intend to calibrate.

2. Separate the synchronized visible and infrared images into two folders, e.g.:

   ```
   data/
     visible/
     thermal/
   ```

3. Create a YAML configuration file (e.g., 'options/dataset/MyDataset.yaml') specifying the paths

4. Change the options in the yaml config file (in `options/mainConf.yaml`) to point toward the desire output folders. 
The Results will be stored in 'your/path/to/results/{experiment_name}'. The given parameters are the default ones we used in our experiments.

```
  name_experiment: your_experiment_name
  
  model:
    Set the index of the target camera: int -> target
    Set the buffer_size for the optimization: int -> buffer_size.
    Set the device to use (cuda/cpu): str -> device.
    train:
        You can adapt the batch_size according your material: int -> batch_size.
        You can choose a Random or a Sequential dataLoader : [random, sequential] -> frame_sampler.
        You can set the weight decay for the training: float -> weight_decay.
        You can set the number of epochs of iteration over the buffer: int -> epochs.
    
  validation:
    You can activate/deactivate visual validation: bool -> visual_validation.
    You can set the frequency of the visual validation update: int -> step_visualize.
    You can also choose the number of images to visualize during the validation step: int -> buffer_size.
    Or you can pick the exact images to visualize by specifying their indices: list -> buffer_idx. (overrid buffer_size)
    
  data:
    Name of the YAML file you created for your dataset in the options/dataset folder: str -> name.
   
  output: Path/to/your/output/folder
```
---

### Running Experiments

Use `Run.py` as the main script to launch the process:

```bash
python Run.py
```
---

## Results & Examples

We provide both **quantitative** and **qualitative** evaluations:

* 📈 Quantitative results can be generated via `test.py` and stored in the `results/` directory.
* 🖼️ Qualitative examples (e.g., overlays, heatmaps) can be saved to `figures/`.

> ⚠️ Example outputs and benchmarks will be added as the project matures.

---

## Citation

If you use this code or method in your work, please cite our paper:

```bibtex
@inproceedings{YourCitationKey,
  title     = {Unsupervised Cameras Calibration for Thermal-Visible Image Registration},
  author    = {…},
  booktitle = {…},
  year      = {2025}
}
```

---

## License

This project is released under the **Unlicense** (public domain).
See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

* Thanks to all contributors and collaborators.
* This work builds upon prior research in camera calibration, image registration, and feature matching.
* We also acknowledge open-source libraries and datasets that made this research possible.

```

Would you like me to also add a **“Quick Start” block** (installation + minimal usage command) at the top so new users can get running in under a minute?
```

