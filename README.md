# HGR-STAR

## 介绍

这是我的深度学习科研项目，HGR-STAR (Raw HD-sEMG Gesture Recognition via StarNet with HD-sEMG Embedding and Label Smoothing) 是借鉴并调整了 StarNet，以适应对 HD-sEMG 信号进行 65 类手势识别的任务。

> StarNet 是微软团队发表在 CVPR 2024 的超强轻量级骨干网络，其论文可从以下地址下载阅读：
> 
> [Rewrite the stars](https://arxiv.org/abs/2403.19967)

本项目使用 C++ 在 Windows 11 平台开发，使用了一个开源的 json 库 [nlohmann/json](https://github.com/nlohmann/json) 以方便配置模型与训练参数，`CMakeLists.txt` 中已经设置了自动下载该三方库依赖，无需手动下载，但如果有再修改本项目的需求，我建议克隆该 json 项目到本地，以便 IDE 能够检索代码。

同时，本项目使用了 PyTorch 的 C++ API [LibTorch](https://pytorch.org/) 来创建模型并编写训练验证代码。其中，[LibTorch](https://pytorch.org/) 的版本为 2.5.1 Release 版本，CUDA 版本为 1.2.4。虽然两者的安装较为繁琐，但网络上有较多教程可供参考，本文不在此赘述，请根据实际需求选择 LibTorch 与 CUDA 的版本。

## 项目结构

### 文件说明

- `.vscode` 文件夹包含本项目使用 Visual Studio Code 开发时，对编辑器的一些配置内容。
- `include` 文件夹包含所有类的声明文件。
- `json` 文件夹包含所有训练任务的 json 文件，通过 `run.bat` 进行自动化的多任务队列训练。
- `src` 文件夹则是这些声明的具体实现，出于快速开发的考虑，并未使用到复杂的模板编程、继承多态等，所以会有大量的重复代码，请见谅。
- `Stanby` 文件夹包含了所有本项目使用过的 json 文件。
- `.gitignore` 文件是本项目使用 git 管理时的文件，记录了忽略那些文件及文件夹的提交推送。
- `CMakeLists.txt` 文件是编译配置文件。
- `LICENSE` 文件则是 MIT 开源许可证。
- `README.en.md` 文件是本文的英文版本。
- `README.md` 文件是本文。
- `run.bat` 文件是进行自动化多任务队列训练的批处理文件。

### 类说明

- `DropPath`：由于 LibTorch 未提供路径丢弃层，故该类是对路径丢弃的实现。
- `Logger`：一个简单的日志类，可以提供基本的文件输出。
- `MultiFeatureDataset`：提供读取数据集功能的类，需要说明的是，**本项目在其他尝试中用到了不同特征，受限于时间，并未删除其中不属于本项目的内容**。
- `StarBlock`：最基本的星运算模块。
- `StarNet2503081200`：使用 **HD-sEMG Embedding 模块**作为输入嵌入的 StarNet，是本文的主推网络。
- `StarNet2503081200TV`：配套前者的训练验证类，`T` 表示训练，`V` 表示验证。
- `StarNet2503081200TVA`：前者的增强版本，引入了标签平滑技术，`A` 表示增强。
- `StarNet2503132330`：使用 **Module 1 模块**作为输入嵌入的 StarNet，是本文的消融对象之一。
- `StarNet2503132330TVA`：配套前者的训练验证类，同样在标签平滑技术的加持下进行训练。
- `StarNet2503141330`：使用 **Module 2 模块**作为输入嵌入的 StarNet，是本文的消融对象之一。
- `StarNet2503141330TVA`：配套前者的训练验证类，同样在标签平滑技术的加持下进行训练。
- `StarNet25101520000`：使用 **基于 STFT 的嵌入模块**作为输入嵌入的 StarNet。
- `StarNet25101520000TVA`：配套前者的训练验证类，在标签平滑技术的加持下进行训练。
- `SummarizeResult`：汇总结果的类。
