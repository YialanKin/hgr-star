#pragma once
#include <torch/torch.h>

class DropPathImpl : public torch::nn::Module {
private:
    double drop_prob = 0.0;

public:
    DropPathImpl(double drop_prob = 0.0);

    torch::Tensor forward(torch::Tensor x);

};
TORCH_MODULE(DropPath);