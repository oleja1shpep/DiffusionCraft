# DiffusionCraft

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains a code for DiffusionCraft project


## Installation

Installation may depend on your task. The general steps are the following:

1. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=3.10

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/3.10/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

2. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

## How To Use

To parse world into schematics run this:

```bash
python3 parse_world.py --world-dir [WORLD_DIR] --output-dir [OUTPUT_DIR] --rx [RADIUS_X] --rz [RADIUS_Z] --n_samples [N_SAMPLES]
```

To parse schematics into tensors:

```bash
python3 schem2tensor.py --schem-dir [SCHEMATICS_DIR] --output-dir [OUTPUT_DIR] --n_samples [N_SAMPLES]
```

To train a VAE, set up `accelerate config` and then run the following command:

```bash
accelerate launch train.py -cn=[CONFIG_NAME] HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

## Credits

This repository is based on a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) and [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repositories.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
