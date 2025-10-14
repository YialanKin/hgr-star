/**
 * @file SummarizeResult.hpp
 * @brief 包含 SummarizeResult 类的声明。
 * @author Rat Kin
 * @date 2025-01-03
 */

#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

/**
 * @brief SummarizeResult 类提供了一个汇总结果的方法。
 */
class SummarizeResult {
public:
    /// @brief 汇总保存目录下的所有结果。
    /// @param res_save 结果保存目录，应包含 20 个受试者，每个受试者五折。
    /// @param sum_save 汇总保存目录。
    static void summarize(std::string res_save, std::string sum_save);
};