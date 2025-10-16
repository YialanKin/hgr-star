# HGR-STAR

## Introduction

This is my deep learning research project. HGR-STAR (Raw HD-sEMG Gesture Recognition via StarNet with HD-sEMG Embedding and Label Smoothing) is an adapted version of StarNet, modified to perform 65-class hand gesture recognition from High-Density Surface Electromyography (HD-sEMG) signals.

> StarNet is an exceptionally lightweight backbone network proposed by Microsoft and published at CVPR 2024. The paper can be downloaded and read here:
> 
> [Rewrite the stars](https://arxiv.org/abs/2403.19967)

This project is developed in C++ on the Windows 11 platform. It uses the open-source JSON library [nlohmann/json](https://github.com/nlohmann/json) for convenient configuration of model and training parameters. The `CMakeLists.txt` file is configured to automatically download this third-party dependency, so manual installation is unnecessary. However, if you plan to modify this project, I recommend cloning the JSON repository locally so that your IDE can properly index and navigate the code.

The project leverages [LibTorch](https://pytorch.org/), the C++ API of PyTorch, to build the model and implement training and validation logic. The [LibTorch](https://pytorch.org/) version used is 2.5.1 Release, with CUDA 1.2.4 support. While installing these dependencies can be complex, numerous online tutorials are available for reference—this document will not elaborate on the setup process. Please select compatible versions of LibTorch and CUDA according to your actual hardware and requirements.

## Project Structure

### File Descriptions

- `.vscode` — Configuration files used when developing this project in Visual Studio Code.
- `include` — Header files containing class declarations.
- `json` — JSON configuration files for training tasks. These are executed in sequence via `run.bat` for automated multi-task queue training.
- `src` — Source code implementing the declarations in include. For rapid development, advanced C++ features such as templates, inheritance, and polymorphism were minimally used, resulting in some duplicated code. Apologies for the lack of abstraction.
- `Stanby` — Archive folder containing all JSON configuration files previously used in this project.
- `.gitignore` — Specifies files and directories to be ignored by Git during version control operations.
- `CMakeLists.txt` — CMake build configuration file.
- `LICENSE` — MIT open-source license.
- `README.en.md` — This document (English version).
- `README.md` — Chinese version of this document.
- `run.bat` — Batch script for automated multi-task queue training.

### Class Descriptions

- `DropPath` — Implements the Stochastic Depth (DropPath) regularization layer, as it is not natively provided in LibTorch.
- `Logger` — A simple logging utility class that supports basic console and file output.
- `MultiFeatureDataset` — A dataset loader class for reading HD-sEMG data. Note: This class was designed to support multiple signal features in earlier experiments. Due to time constraints, legacy code for unused features has not been removed.
- `StarBlock` — The fundamental building block of StarNet, implementing the "star operation".
- `StarNet2503081200` — The primary network of this work: a StarNet variant using the HD-sEMG Embedding module as its input embedding.
- `StarNet2503081200TV` — Training and validation wrapper for the above model. T denotes Training, V denotes Validation.
- `StarNet2503081200TVA` — Enhanced version of the above, incorporating Label Smoothing for improved generalization. A stands for Augmentation/Enhancement.
- `StarNet2503132330` — A StarNet variant using Module 1 as input embedding, used as one of the ablation study targets.
- `StarNet2503132330TVA` — Corresponding training and validation class, also trained with label smoothing.
- `StarNet2503141330` — A StarNet variant using Module 2 as input embedding, used as another ablation study target.
- `StarNet2503141330TVA` — Corresponding training and validation class, also trained with label smoothing.
- `StarNet25101520000` — StarNet with a STFT-based embedding module as input embedding.
- `StarNet25101520000TVA` — The corresponding training and validation class for the former, trained with the aid of label smoothing technique.
- `SummarizeResult` — A utility class for summarizing and organizing experimental results.