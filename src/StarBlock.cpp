#include "StarBlock.hpp"

StarBlockImpl::StarBlockImpl(StarBlockOptions& config) {
    dw1 = register_module("DW1", torch::nn::Conv2d(config.dw1_config));
    pw1 = register_module("PW1", torch::nn::Conv2d(config.pw1_config));
    bn1 = register_module("BN1", torch::nn::BatchNorm2d(config.bn1_config));
    fc1 = register_module("FC1", torch::nn::Linear(config.fc1_config));
    act = register_module("ACT", torch::nn::ReLU6());
    fc2 = register_module("FC2", torch::nn::Linear(config.fc2_config));
    fc3 = register_module("FC3", torch::nn::Linear(config.fc3_config));
    bn2 = register_module("BN2", torch::nn::BatchNorm2d(config.bn2_config));
    dw2 = register_module("DW2", torch::nn::Conv2d(config.dw2_config));
    pw2 = register_module("PW2", torch::nn::Conv2d(config.pw2_config));
    drop = register_module("DropPath", DropPath(config.DropPathRate()));
    return;
}

torch::Tensor StarBlockImpl::forward(torch::Tensor x) {
    torch::Tensor input = x.clone();
    x = dw1->forward(x);
    x = pw1->forward(x);
    x = bn1->forward(x);
    x = x.permute({0, 2, 3, 1});
    auto bh1 = act->forward(fc1->forward(x)).permute({0, 3, 1, 2});
    auto bh2 = fc2->forward(x).permute({0, 3, 1, 2});
    x = bh1 * bh2;
    x = x.permute({0, 2, 3, 1});
    x = fc3->forward(x).permute({0, 3, 1, 2});
    x = bn2->forward(x);
    x = dw2->forward(x);
    x = pw2->forward(x);
    return input + drop->forward(x);
}


void StarBlockOptions::update(void) {
    dw1_config = torch::nn::Conv2dOptions(dim, dim, ks).groups(dim).padding({ks / 2, ks / 2});
    pw1_config = torch::nn::Conv2dOptions(dim, dim, 1).stride(1);
    bn1_config = torch::nn::BatchNorm2dOptions(dim);
    fc1_config = torch::nn::LinearOptions(dim, hidden_ratio * dim);
    fc2_config = torch::nn::LinearOptions(dim, hidden_ratio * dim);
    fc3_config = torch::nn::LinearOptions(hidden_ratio * dim, dim);
    bn2_config = torch::nn::BatchNorm2dOptions(dim);
    dw2_config = torch::nn::Conv2dOptions(dim, dim, ks).groups(dim).padding({ks / 2, ks / 2});
    pw2_config = torch::nn::Conv2dOptions(dim, dim, 1).stride(1);
    return;
}

double StarBlockOptions::DropPathRate(void) {
    return drop_rate;
}

StarBlockOptions& StarBlockOptions::setDim(int value) {
    this->dim = value;
    update();
    return *this;
}

StarBlockOptions& StarBlockOptions::setDWConvKS(int value) {
    this->ks = value;
    update();
    return *this;
}

StarBlockOptions& StarBlockOptions::setHiddenRatio(int value) {
    this->hidden_ratio = value;
    update();
    return *this;
}

StarBlockOptions& StarBlockOptions::setDropRate(double value) {
    this->drop_rate = value;
    return *this;
}