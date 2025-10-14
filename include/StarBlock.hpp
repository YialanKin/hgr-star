#pragma once
#include <torch/torch.h>
#include "DropPath.hpp"

class StarBlockOptions {
private:
    int dim = 32;
    int ks = 7;
    int hidden_ratio = 4;
    double drop_rate = 0.0;

    void update(void);

public:
    torch::nn::Conv2dOptions dw1_config = torch::nn::Conv2dOptions(32, 32, 7).groups(32).padding({7 / 2, 7 / 2});
    torch::nn::Conv2dOptions pw1_config = torch::nn::Conv2dOptions(32, 32, 1).stride(1);
    torch::nn::BatchNorm2dOptions bn1_config = torch::nn::BatchNorm2dOptions(32);
    torch::nn::LinearOptions fc1_config = torch::nn::LinearOptions(32, 4 * 32);
    torch::nn::LinearOptions fc2_config = torch::nn::LinearOptions(32, 4 * 32);
    torch::nn::LinearOptions fc3_config = torch::nn::LinearOptions(4 * 32, 32);
    torch::nn::BatchNorm2dOptions bn2_config = torch::nn::BatchNorm2dOptions(32);
    torch::nn::Conv2dOptions dw2_config = torch::nn::Conv2dOptions(32, 32, 7).groups(32).padding({7 / 2, 7 / 2});
    torch::nn::Conv2dOptions pw2_config = torch::nn::Conv2dOptions(32, 32, 1).stride(1);
    double DropPathRate(void);

    StarBlockOptions() = default;

    StarBlockOptions& setDim(int value);
    StarBlockOptions& setDWConvKS(int value);
    StarBlockOptions& setHiddenRatio(int value);
    StarBlockOptions& setDropRate(double value);
};

class StarBlockImpl : public torch::nn::Module {
private:
    torch::nn::Conv2d dw1 = nullptr;
    torch::nn::Conv2d pw1 = nullptr;
    torch::nn::BatchNorm2d bn1 = nullptr;
    torch::nn::Linear fc1 = nullptr;
    torch::nn::ReLU6 act = nullptr;
    torch::nn::Linear fc2 = nullptr;
    torch::nn::Linear fc3 = nullptr;
    torch::nn::BatchNorm2d bn2 = nullptr;
    torch::nn::Conv2d dw2 = nullptr;
    torch::nn::Conv2d pw2 = nullptr;
    DropPath drop = nullptr;

public:
    StarBlockImpl(StarBlockOptions& config);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(StarBlock);