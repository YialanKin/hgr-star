#pragma once
#include "StarBlock.hpp"

class StarNet2503132330Options {
private:
    int in_channels = 512;
    int upsample_size = 224;
    int dechannels = 3;
    int downsample_ratio = 4;
    int embed = 32;
    int conv_ks = 3;
    int conv_stride = 2;
    double drop_rate = 0.1;
    int num_classes = 65;

    std::vector<int> depths{3, 3, 12, 5};

    void update(void);

public:
    torch::nn::UpsampleOptions pre_upsample_config = torch::nn::UpsampleOptions().size(std::vector<int64_t>({224, 224})).mode(torch::kBilinear).align_corners(false);
    torch::nn::Conv2dOptions pre_dechannel_config = torch::nn::Conv2dOptions(512, 3, 1);
    torch::nn::Conv2dOptions pre_conv_config = torch::nn::Conv2dOptions(3, 24, 3).stride(7).padding(1);
    torch::nn::BatchNorm2dOptions pre_bn_config = torch::nn::BatchNorm2dOptions(32);

    
    torch::nn::Conv2dOptions stage1_conv_config = torch::nn::Conv2dOptions(32, 32, 3).stride(2).padding(1);
    std::vector<StarBlockOptions> stage1_block_config = std::vector<StarBlockOptions>(3);

    torch::nn::Conv2dOptions stage2_conv_config = torch::nn::Conv2dOptions(32, 32, 3).stride(2).padding(1);
    std::vector<StarBlockOptions> stage2_block_config = std::vector<StarBlockOptions>(3);

    torch::nn::Conv2dOptions stage3_conv_config = torch::nn::Conv2dOptions(32, 32, 3).stride(2).padding(1);
    std::vector<StarBlockOptions> stage3_block_config = std::vector<StarBlockOptions>(12);

    torch::nn::Conv2dOptions stage4_conv_config = torch::nn::Conv2dOptions(32, 32, 3).stride(2).padding(1);
    std::vector<StarBlockOptions> stage4_block_config = std::vector<StarBlockOptions>(5);

    torch::nn::BatchNorm2dOptions norm_config = torch::nn::BatchNorm2dOptions(32);
    torch::nn::AdaptiveAvgPool2dOptions avgpool_config = torch::nn::AdaptiveAvgPool2dOptions(1);
    torch::nn::LinearOptions head_config = torch::nn::LinearOptions(32, 65);

    StarNet2503132330Options(void) {
        update();
        return;
    }

    int Depth(int stage);
    StarNet2503132330Options& setInChannels(int value);
    StarNet2503132330Options& setUpsampleSize(int value);
    StarNet2503132330Options& setDechannels(int value);
    StarNet2503132330Options& setDownsampleRatio(int value);
    StarNet2503132330Options& setEmbed(int value);
    StarNet2503132330Options& setConvKs(int value);
    StarNet2503132330Options& setConvStride(int value);
    StarNet2503132330Options& setDropRate(double value);
    StarNet2503132330Options& setNumClasses(int value);
    StarNet2503132330Options& setDepths(std::vector<int> value);

};

class StarNet2503132330Impl : public torch::nn::Module {
private:
    // Stem
    torch::nn::Upsample pre_upsample = nullptr;
    torch::nn::Conv2d pre_dechannel = nullptr;
    torch::nn::Conv2d pre_conv = nullptr;
    torch::nn::BatchNorm2d pre_bn = nullptr;
    torch::nn::ReLU6 pre_act = nullptr;

    // Stages
    torch::nn::Conv2d stage1_conv = nullptr;
    torch::nn::Sequential stage1_blocks = nullptr;
    torch::nn::Conv2d stage2_conv = nullptr;
    torch::nn::Sequential stage2_blocks = nullptr;
    torch::nn::Conv2d stage3_conv = nullptr;
    torch::nn::Sequential stage3_blocks = nullptr;
    torch::nn::Conv2d stage4_conv = nullptr;
    torch::nn::Sequential stage4_blocks = nullptr;

    // Head
    torch::nn::BatchNorm2d norm = nullptr;
    torch::nn::AdaptiveAvgPool2d avgpool = nullptr;
    torch::nn::Linear head = nullptr;
    
public:
    StarNet2503132330Impl(StarNet2503132330Options model_config);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(StarNet2503132330);