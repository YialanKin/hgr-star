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
- `StarNet2503081200` — StarNet variant using a **ConvTranspose-based HD-sEMG embedding module**.
- `StarNet2503081200TV` — Training and validation wrapper for the above model. T denotes Training, V denotes Validation.
- `StarNet2503081200TVA` — Enhanced version of the above, incorporating **label smoothing** for improved generalization. A stands for Augmentation/Enhancement.
- `StarNet2503132330` — StarNet using an **Upsample + Conv + Conv-based HD-sEMG embedding module**.
- `StarNet2503132330TVA` — Corresponding training and validation class, also trained with **label smoothing**.
- `StarNet2503141330` — StarNet using an **Upsample + Conv-based HD-sEMG embedding module**.
- `StarNet2503141330TVA` — Corresponding training and validation class, also trained with **label smoothing**.
- `StarNet2510152000` — StarNet using an **STFT + Conv-based HD-sEMG embedding module**.
- `StarNet2510152000TV` — Training and validation wrapper for the above model. T denotes Training, V denotes Validation.
- `StarNet2510152000TVA` — The corresponding training and validation class for the former, trained with the aid of **label smoothing** technique.
- `StarNet2510161140`：A refined version of the STFT-based embedding module.
- `StarNet2510161140TVA`：The corresponding training and validation class for the above model, trained with **label smoothing**.
- `SummarizeResult` — A utility class for summarizing and organizing experimental results.

## Deployment Guide

### Hardware Configuration

The last successful run of this project was performed on the following hardware:

- **CPU:** 13th Gen Intel(R) Core(TM) i7-13700KF (3.40 GHz)
- **GPU:** ROG STRIX RTX4080S O16G GAMING
- **RAM:** Asgard DDR5 32 GB(16 GB x 2) 6800 MHz CL38
- **Motherboard:** ROG STRIX B760-G GAMING WIFI
- **Storage:**
  - **SSD:** ZHITAI TiPlus7100 SSD 512 GB
  - **SSD:** ZHITAI TiPlus7100 SSD 2048 GB
- **Power Supply Unit (PSU):** Seasonic Focus GX-850

### Software Configuration

The software environment used during the last run:

- **System:**
  - **Edition:** Windows 11 Home
  - **Version:** 25H2
  - **Installed on:** 2025/1/25
  - **OS build:** 26200.7623
- **GPU Driver:** 581.57
- **NVIDIA (R) Cuda Compiler Driver (NVCC):** release 12.4, V12.4.99
- **LibTorch:** 2.5.1+cu124
- **C/C++ Compiler:** Microsoft (R) C/C++ Optimizing Compiler Version 19.44.35222 for x64
- **CMake:** 3.30.1
- **MATLAB:** R2018b
- **Git:** 2.51.0.windows.2
- **Python:** 3.7.3

### Software Installation

While the installation order may not strictly matter, we recommend the following sequence.

#### C/C++ Compiler

On Windows, we strongly recommend installing [Microsoft Visual Studio](https://visualstudio.microsoft.com/zh-hans/) to obtain the best-supported C++ toolchain.

> ⚠️ **Note:** As of **January 18, 2026, CUDA Toolkit 12.4** and **CUDA Toolkit 13.1 (latest)** do **not** support **Visual Studio 2026** in their **Visual Studio Integration**. Therefore, **install Visual Studio 2022** instead.

> 🕰 **Local Time:** January 18, 2026, 03:01:50

1. Download **VisualStudioSetup.exe** from [Microsoft Visual Studio](https://visualstudio.microsoft.com/zh-hans/).
2. Run the installer to install the **Visual Studio Installer**.
3. Close the installer after it launches.
4. In Command Prompt, run:
    ```
    winget show --versions "Microsoft.VisualStudio.2022.Community"
    ```
    to list available versions.  
    ![Visual Studio Installer - Install Visual Studio 2026 - image 1](.\readmeImages\image32.png)
5. Install a compatible version (e.g., 17.14.24):
    ```
    winget install -v 17.14.24 "Microsoft.VisualStudio.2022.Community"
    ```
    The installer will launch automatically.
6. Reopen **Visual Studio Installer**; you should now see **Visual Studio 2022 Community**.  
    ![Visual Studio Installer - Install Visual Studio 2026 - image 2](.\readmeImages\image33.png)
7. Click **Modify**, ensure **Desktop development with C++** is selected under **Workloads**, then install.  
    ![Visual Studio Installer - Install Visual Studio 2026 - image 3](.\readmeImages\image34.png)
8. Restart your computer.
9. Add necessary paths to environment variables.  
    ![Visual Studio Installer - Install Visual Studio 2026 - image 4](.\readmeImages\image35.png)
10. Verify installation by running `cl` in Command Prompt.  
    ![Visual Studio Installer - Install Visual Studio 2026 - image 5](.\readmeImages\image36.png)

#### GPU Driver

GPU drivers are typically backward-compatible with CUDA versions. Use the [NVIDIA App](https://www.nvidia.cn/software/nvidia-app/) or another method to update your driver to ensure support for [CUDA 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive).

Steps used in our environment:

> 🕰 **Local Time:** January 17, 2026, 18:13:07

1. Open **NVIDIA App**.  
    ![NVIDIA App - Update Driver - image 1](.\readmeImages\image8.png)
2. In the **Drivers** tab, **Download** the latest driver (we used GeForce Game Ready).  
    ![NVIDIA App - Update Driver - image 2](.\readmeImages\image9.png)
3. **Install** the downloaded driver (we chose **Quick Installation**).  
    ![NVIDIA App - Update Driver - image 3](.\readmeImages\image10.png)
4. Restart your computer.  
    ![NVIDIA App - Update Driver - image 4](.\readmeImages\image11.png)
5. Verify with `nvidia-smi`. Example output: `Driver Version: 591.74` and `CUDA Version: 13.1` indicate compatibility with CUDA 12.4.  
    ![NVIDIA App - Update Driver - image 5](.\readmeImages\image14.png)

#### CUDA Toolkit

This project was developed with [CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local). We recommend using this exact version to ensure compatibility.

> ⚠️ **Note:** Our system already had CUDA 12.4 installed; uninstalling CUDA is complex, so only key steps are provided.

> 🕰 **Local Time:** January 17, 2026, 19:34:47

1. On the [CUDA 12.4 download page](https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local), select your platform and download the installer.  
    ![CUDA - Install - image 1](.\readmeImages\image15.png)
2. Run **cuda_12.4.0_551.61_windows.exe**.  
    ![CUDA - Install - image 2](.\readmeImages\image16.png)
3. Install using **Express** mode.  
    ![CUDA - Install - image 3](.\readmeImages\image17.png)
    > ⚠️ **Critical Note:** The CUDA 12.4 installer **does not include NVTX (NVIDIA Tools Extension Library)**, which is required by LibTorch. To resolve this, install NVTX separately from CUDA 11.8:
4. Go to the [CUDA Toolkit 11.8 download page](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local) and download the installer.  
    ![CUDA - Install - image 4](.\readmeImages\image18.png)
5. Run `cuda_11.8.0_522.06_windows.exe`.
6. Choose **Custom** installation and **only select Nsight NVTX**.  
    ![CUDA - Install - image 5](.\readmeImages\image19.png)
7. Ensure the following environment variables are set:
   - `CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`
   - `CUDA_PATH_V12_4=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`
   - `NVTOOLSEXT_PATH=C:\Program Files\NVIDIA Corporation\NvToolsExt\`
   - `PATH`:
     - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin`
     - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp`

    ![CUDA - Install - image 6](.\readmeImages\image20.png)  
    ![CUDA - Install - image 7](.\readmeImages\image21.png)
8. Verify with `nvcc --version`.  
    ![CUDA - Install - image 8](.\readmeImages\image22.png)

#### LibTorch

LibTorch versions are tied to CUDA versions. While minor version differences may work, we recommend **LibTorch 2.5.1+cu124 (Release)**.

Installation is straightforward: download and extract.

> 🕰 **Local Time:** January 17, 2026, 20:42:46

1. Visit [https://download.pytorch.org/libtorch/cu124/](https://download.pytorch.org/libtorch/cu124/).
    - **libtorch-win-shared-with-deps-2.x.x%2Bcu124.zip:** Windows Release
    - **libtorch-win-shared-with-deps-debug-2.x.x%2Bcu124.zip:** Windows Debug

    We recommend **Release** for performance.  
    ![LibTorch - Install - image 1](.\readmeImages\image23.png)
2. Download [LibTorch 2.5.1+cu124 Release](https://download.pytorch.org/libtorch/cu124/libtorch-win-shared-with-deps-2.5.1%2Bcu124.zip).
3. Extract to a disk with sufficient space. Rename the folder for clarity (e.g., **libtorch251cu124**).  
    ![LibTorch - Install - image 2](.\readmeImages\image24.png)

#### CMake

> 🕰 **Local Time:** January 17, 2026, 20:50:52

> ⚠️ **Note:** Very new CMake versions (e.g., hypothetical 4.2.1) may fail compatibility checks with projects requiring CMake < 3.5. We used **CMake 3.30.1**.

1. Download **cmake-3.30.1-windows-x86_64.msi** from [CMake’s archive]((https://cmake.org/files/v3.30/)).  
    ![CMake - Install - image 1](.\readmeImages\image25.png)
2. Run the **.msi** installer.
3. Ensure CMake is in your **PATH**.  
    ![CMake - Install - image 2](.\readmeImages\image26.png)
4. Verify with `cmake --version`.  
    ![CMake - Install - image 3](.\readmeImages\image31.png)

#### MATLAB

Since the dataset is in **.mat** format, **MATLAB R2018b** is used for basic preprocessing (splitting into 5-fold cross-validation sets).

As MATLAB is commercial software, we do not provide installation instructions. **Use R2018b** to avoid compatibility issues with newer versions.

### Project Deployment

#### Downloading the Open Dataset

The dataset is described in:

[A database of high-density surface electromyogram signals comprising 65 isometric hand gestures](https://www.nature.com/articles/s41597-021-00843-9#Sec8)

Download all **20 subject files** (total ~131 GB) from [figshare](https://springernature.figshare.com/collections/A_database_of_high-density_surface_electromyogram_signals_comprising_65_hand_gestures_performed_in_an_isometric_manner/5090861).

> ⚠️ **Note:** Ensure sufficient disk space.

#### Preparing the 5-Fold Dataset

> ⚠️ **Note:** The following steps require this project to be cloned locally.

1. Edit `.\matlabScripts\generate_hdsEMG_windowed_samples.m`:
    - Set `raw_data_root_dir` to your downloaded dataset root.
    - Set `output_root_dir` to your desired output path.

    ![Generate HD-sEMG windowed samples - image 1](.\readmeImages\image28.png)
2. Run the script in MATLAB until all data is processed.

#### Building HGR-STAR

1. Open Command Prompt in the project root.
2. Create and enter a build directory:
    ```
    mkdir build
    cd build
    ```
3. Configure with CMake (replace path with your LibTorch location):
    ```
    cmake -DCMAKE_PREFIX_PATH="D:\libtorch\libtorch251cu124" ..
    ```
    > ⚠️ Ensure internet access to download **nlohmann/json**.

    ![Build - image 1](.\readmeImages\image37.png)
4. Build in Release mode:
    ```
    cmake --build . --config release
    ```
    > ⚠️ Use `--config debug` if using a Debug version of LibTorch.

    ![Build - image 2](.\readmeImages\image38.png)

#### Training and Validation

1. Copy desired JSON config(s) from `.\Standby\` to `.\json\`.  
    ![Train - image 1](.\readmeImages\image39.png)
2. Edit each JSON file:
    - Set `rootOrig` to your `output_root_dir` from MATLAB preprocessing.
    - Set `savePath` to where results (models, logs) will be saved.
  
    ![Train - image 2](.\readmeImages\image40.png)
3. Run the training queue from the project root:
    ```
    .\run.bat
    ```
    ![Train - image 3](.\readmeImages\image41.png)

    ![Train - image 4](.\readmeImages\image42.png)
4. Inspect results in `savePath`.  
    ![Train - image 4](.\readmeImages\image43.png)