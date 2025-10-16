#include "StarNet2510152000.hpp"

StarNet2510152000Impl::StarNet2510152000Impl(StarNet2510152000Options model_config) {

    pre_conv = register_module("PreConv", torch::nn::Conv2d(model_config.pre_conv_config));
    pre_bn = register_module("PreBN", torch::nn::BatchNorm2d(model_config.pre_bn_config));
    pre_act = register_module("PreAct", torch::nn::ReLU6());

    stage1_conv = register_module("Stage1Conv", torch::nn::Conv2d(model_config.stage1_conv_config));
    stage1_blocks = torch::nn::Sequential();
    for (int i = 0; i < model_config.Depth(1); ++i) {
        stage1_blocks->push_back(StarBlock(model_config.stage1_block_config[i]));
    }
    stage1_blocks = register_module("Stage1Blocks", stage1_blocks);

    stage2_conv = register_module("Stage2Conv", torch::nn::Conv2d(model_config.stage2_conv_config));
    stage2_blocks = torch::nn::Sequential();
    for (int i = 0; i < model_config.Depth(2); ++i) {
        stage2_blocks->push_back(StarBlock(model_config.stage2_block_config[i]));
    }
    stage2_blocks = register_module("Stage2Blocks", stage2_blocks);

    stage3_conv = register_module("Stage3Conv", torch::nn::Conv2d(model_config.stage3_conv_config));
    stage3_blocks = torch::nn::Sequential();
    for (int i = 0; i < model_config.Depth(3); ++i) {
        stage3_blocks->push_back(StarBlock(model_config.stage3_block_config[i]));
    }
    stage3_blocks = register_module("Stage3Blocks", stage3_blocks);

    stage4_conv = register_module("Stage4Conv", torch::nn::Conv2d(model_config.stage4_conv_config));
    stage4_blocks = torch::nn::Sequential();
    for (int i = 0; i < model_config.Depth(4); ++i) {
        stage4_blocks->push_back(StarBlock(model_config.stage4_block_config[i]));
    }
    stage4_blocks = register_module("Stage4Blocks", stage4_blocks);

    norm = register_module("Norm", torch::nn::BatchNorm2d(model_config.norm_config));
    avgpool = register_module("GAP", torch::nn::AdaptiveAvgPool2d(model_config.avgpool_config));
    head = register_module("Head", torch::nn::Linear(model_config.head_config));
    return;
}

torch::Tensor StarNet2510152000Impl::forward(torch::Tensor x) {
    int b = x.size(0);
    int c = x.size(1);
    int d = x.size(2);
    int h = x.size(3);
    int w = x.size(4);

    // Embedding
    x = x.permute({0, 1, 3, 4, 2});     // [b, c, d, h, w] -> [b, c, h, w, d]
    x = x.reshape({b * c * h * w, d});  // [b, c, h, w, d] -> [b * c * h * w, d]
    x = torch::stft(                    // [b * c * h * w, d] -> [b * c * h * w, 33, 24]
        x,
        /*n_fft=*/64,
        /*hop_length=*/19,
        /*win_length=*/64,
        /*window=*/window.to(x.device()),
        /*normalized=*/false,
        /*onesided=*/true,
        /*return_complex=*/true
    );
    x = torch::abs(x);  // 取幅度谱
    x = x.index({torch::indexing::Slice(), torch::indexing::Slice(1, 33), torch::indexing::Slice()});  // 去除零频分量 [b * c * h * w, 33, 24] -> [b * c * h * w, 32, 24]
    int f = x.size(1);
    int t = x.size(2);
    x = x.reshape({b * c, h * w, f, t});    // [b * c * h * w, 32, 24] -> [b * c, h * w, 32, 24]
    x = pre_conv->forward(x);               // [b * c, h * w = 64, 32, 24] -> [b * c, h * w = 32, 32, 24]
    x = x.permute({0, 3, 2, 1});
    x = pre_bn->forward(x);
    x = pre_act->forward(x);

    x = stage1_conv->forward(x);
    x = stage1_blocks->forward(x);
    x = stage2_conv->forward(x);
    x = stage2_blocks->forward(x);
    x = stage3_conv->forward(x);
    x = stage3_blocks->forward(x);
    x = stage4_conv->forward(x);
    x = stage4_blocks->forward(x);

    x = norm->forward(x);
    x = avgpool->forward(x);
    x = x.reshape({b, c * x.size(1)});
    x = torch::flatten(x, 1);
    x = head->forward(x);
    return x;
}


void StarNet2510152000Options::update(void) {
    // pre_conv_config = torch::nn::ConvTranspose2dOptions(in_channels, embed, 4).stride(4).padding(0);
    pre_bn_config = torch::nn::BatchNorm2dOptions(embed);

    int depths_sum = 0;
    for (int i = 0; i < 4; ++i) {
        depths_sum += depths[i];
    }
    torch::Tensor dpr_tensor = torch::linspace(0, drop_rate, depths_sum);
    std::vector<double> dpr(dpr_tensor.numel());
    for (int i = 0; i < dpr.size(); ++i) {
        dpr[i] = dpr_tensor[i].item<double>();
    }

    stage1_conv_config = torch::nn::Conv2dOptions(embed, embed, conv_ks).stride(conv_stride).padding(1);
    stage1_block_config = std::vector<StarBlockOptions>(depths[0]);
    for (int i = 0; i < depths[0]; ++i) {
        stage1_block_config[i].setDim(embed);
        stage1_block_config[i].setDropRate(dpr[i]);
    }

    stage2_conv_config = torch::nn::Conv2dOptions(embed, 2 * embed, conv_ks).stride(conv_stride).padding(1);
    stage2_block_config = std::vector<StarBlockOptions>(depths[1]);
    for (int i = 0; i < depths[1]; ++i) {
        stage2_block_config[i].setDim(2 * embed);
        stage2_block_config[i].setDropRate(dpr[depths[0] + i]);
    }

    stage3_conv_config = torch::nn::Conv2dOptions(2 * embed, 4 * embed, conv_ks).stride(conv_stride).padding(1);
    stage3_block_config = std::vector<StarBlockOptions>(depths[2]);
    for (int i = 0; i < depths[2]; ++i) {
        stage3_block_config[i].setDim(4 * embed);
        stage3_block_config[i].setDropRate(dpr[depths[1] + i]);
    }

    stage4_conv_config = torch::nn::Conv2dOptions(4 * embed, 8 * embed, conv_ks).stride(conv_stride).padding(1);
    stage4_block_config = std::vector<StarBlockOptions>(depths[3]);
    for (int i = 0; i < depths[3]; ++i) {
        stage4_block_config[i].setDim(8 * embed);
        stage4_block_config[i].setDropRate(dpr[depths[2] + i]);
    }

    norm_config = torch::nn::BatchNorm2dOptions(8 * embed);
    avgpool_config = torch::nn::AdaptiveAvgPool2dOptions(1);
    head_config = torch::nn::LinearOptions(2 * 8 * embed, num_classes);
    return;
}

int StarNet2510152000Options::Depth(int stage) {
    return depths[stage - 1];
}

StarNet2510152000Options& StarNet2510152000Options::setInChannels(int value) {
    in_channels = value;
    update();
    return *this;
}

StarNet2510152000Options& StarNet2510152000Options::setEmbed(int value) {
    embed = value;
    update();
    return *this;
}

StarNet2510152000Options& StarNet2510152000Options::setConvKs(int value) {
    conv_ks = value;
    update();
    return *this;
}

StarNet2510152000Options& StarNet2510152000Options::setConvStride(int value) {
    conv_stride = value;
    update();
    return *this;
}

StarNet2510152000Options& StarNet2510152000Options::setDropRate(double value) {
    drop_rate = value;
    update();
    return *this;
}

StarNet2510152000Options& StarNet2510152000Options::setNumClasses(int value) {
    num_classes = value;
    update();
    return *this;
}

StarNet2510152000Options& StarNet2510152000Options::setDepths(std::vector<int> value) {
    depths = value;
    update();
    return *this;
}