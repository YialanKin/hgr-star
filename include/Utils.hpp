#pragma once
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

class Utils {
public:
    /// @brief 汇总保存目录下的所有结果。
    /// @param res_save 结果保存目录，应包含 20 个受试者，每个受试者五折。
    /// @param sum_save 汇总保存目录。
    static void summarize(std::string res_save, std::string sum_save);

    /// @brief 计算模型总参数量
    /// @param model 模型
    /// @return 参数量
    static int64_t countParameters(const std::shared_ptr<torch::nn::Module> model);
};