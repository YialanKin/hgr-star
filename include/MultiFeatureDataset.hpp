/**
 * @file MultiFeatureDataset.hpp
 * @brief 包含 MultiFeatureDataset 类、配置结构体及构建器类的声明。
 * @author Rat Kin
 * @date 2025-03-02
 */

#pragma once
#include <torch/torch.h>
#include <filesystem>
#include <fstream>

struct MultiFeatureDatasetConfig {
    /** 原始特征五折目录列表 */
    std::vector<std::string> paths_orig;

    /** 原始特征的通道数 */
    int nchannels_orig = 2;

    /** 原始特征的深度 */
    int depth_orig = 512;
    
    /** 原始特征的高度 */
    int height_orig = 8;

    /** 原始特征的宽度 */
    int width_orig = 8;

    /** 包络特征五折目录列表 */
    std::vector<std::string> paths_enve;

    /** 包络特征的通道数 */
    int nchannels_enve = 2;

    /** 包络特征的深度 */
    int depth_enve = 512;
    
    /** 包络特征的高度 */
    int height_enve = 8;

    /** 包络特征的宽度 */
    int width_enve = 8;

    /** MUAP 峰峰值图特征五折目录列表 */
    std::vector<std::string> paths_muapp2p;

    /** MUAP 峰峰值图特征的通道数 */
    int nchannels_muapp2p = 2;

    /** MUAP 峰峰值图特征的深度 */
    int depth_muapp2p = 8;
    
    /** MUAP 峰峰值图特征的高度 */
    int height_muapp2p = 8;

    /** MUAP 峰峰值图特征的宽度 */
    int width_muapp2p = 8;

    /** 常见特征 (MAV, ZCR, WL, Willison Amplitude) 五折目录列表 */
    std::vector<std::string> paths_comm;

    /** 常见特征的通道数 */
    int nchannels_comm = 2;

    /** 常见特征的深度 */
    int depth_comm = 8;
    
    /** 常见特征的高度 */
    int height_comm = 8;

    /** 常见特征的宽度 */
    int width_comm = 8;

};

class MultiFeatureDataset : public torch::data::Dataset<MultiFeatureDataset> {
public:
    /// @brief MultiFeatureDataset 的构造函数，接受配置结构体并根据其中的参数完成数据集的初始化。
    /// @param config MultiFeatureDataset 的配置结构体。
    MultiFeatureDataset(const MultiFeatureDatasetConfig& config);

    /// @brief 获取数据集中样本的数量。
    /// @param  
    /// @return 数据集中样本的数量。
    torch::optional<size_t> size(void) const override;

    /// @brief 获取指定下标的样本。
    /// @param index 指定下标。
    /// @return 样本的数据与标签。
    torch::data::Example<> get(size_t index) override;


private:
    /** 原始特征的文件列表 */
    std::vector<std::string> files_orig;

    /** 原始特征的通道数 */
    int nchannels_orig;

    /** 原始特征的深度 */
    int depth_orig;
    
    /** 原始特征的高度 */
    int height_orig;

    /** 原始特征的宽度 */
    int width_orig;

    /** 原始特征的数据长度 */
    int datalen_orig;

    /** 包络特征的文件列表 */
    std::vector<std::string> files_enve;

    /** 包络特征的通道数 */
    int nchannels_enve;

    /** 包络特征的深度 */
    int depth_enve;
    
    /** 包络特征的高度 */
    int height_enve;

    /** 包络特征的宽度 */
    int width_enve;

    /** 包络特征的数据长度 */
    int datalen_enve;

    /** MUAP 峰峰值特征的文件列表 */
    std::vector<std::string> files_muapp2p;

    /** MUAP 峰峰值图特征的通道数 */
    int nchannels_muapp2p;

    /** MUAP 峰峰值图特征的深度 */
    int depth_muapp2p;
    
    /** MUAP 峰峰值图特征的高度 */
    int height_muapp2p;

    /** MUAP 峰峰值图特征的宽度 */
    int width_muapp2p;

    /** MUAP 峰峰值图特征的数据长度 */
    int datalen_muapp2p;

    /** 常见特征的文件列表 */
    std::vector<std::string> files_comm;

    /** 常见特征的通道数 */
    int nchannels_comm;

    /** 常见特征的深度 */
    int depth_comm;
    
    /** 常见特征的高度 */
    int height_comm;

    /** 常见特征的宽度 */
    int width_comm;
    
    /** 常见特征的数据长度 */
    int datalen_comm;

    /** 标签列表 */
    std::vector<int> classes;

};

class MultiFeatureDatasetBuilder {
public:
    /// @brief 添加一个原始特征的折目录。
    /// @param value 折目录。
    /// @return 当前构建器对象，支持链式调用。
    MultiFeatureDatasetBuilder& addPath_Orig(std::string value);

    /// @brief 添加一个包络特征的折目录。
    /// @param value 折目录。
    /// @return 当前构建器对象，支持链式调用。
    MultiFeatureDatasetBuilder& addPath_Enve(std::string value);

    /// @brief 添加一个 MUAP 峰峰值特征的折目录。
    /// @param value 折目录。
    /// @return 当前构建器对象，支持链式调用。
    MultiFeatureDatasetBuilder& addPath_MUAPP2P(std::string value);

    /// @brief 添加一个常见特征的折目录。
    /// @param value 折目录。
    /// @return 当前构建器对象，支持链式调用。
    MultiFeatureDatasetBuilder& addPath_Comm(std::string value);

    /// @brief 清空折目录。
    /// @param 
    /// @return 当前构建器对象，支持链式调用。
    MultiFeatureDatasetBuilder& clearPaths(void);

    /// @brief 设置原始特征的通道数
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    MultiFeatureDatasetBuilder& setNChannels_Orig(int value);

    /// @brief 设置原始特征的深度
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    MultiFeatureDatasetBuilder& setDepth_Orig(int value);

    /// @brief 设置原始特征的高度
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    MultiFeatureDatasetBuilder& setHeight_Orig(int value);

    /// @brief 设置原始特征的宽度
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    MultiFeatureDatasetBuilder& setWidth_Orig(int value);

    /// @brief 设置包络特征的通道数
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    MultiFeatureDatasetBuilder& setNChannels_Enve(int value);

    /// @brief 设置包络特征的深度
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    MultiFeatureDatasetBuilder& setDepth_Enve(int value);

    /// @brief 设置包络特征的高度
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    MultiFeatureDatasetBuilder& setHeight_Enve(int value);

    /// @brief 设置包络特征的宽度
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    MultiFeatureDatasetBuilder& setWidth_Enve(int value);

    /// @brief 设置 MUAP 峰峰值图特征的通道数
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    MultiFeatureDatasetBuilder& setNChannels_MUAPP2P(int value);

    /// @brief 设置 MUAP 峰峰值图特征的深度
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    MultiFeatureDatasetBuilder& setDepth_MUAPP2P(int value);

    /// @brief 设置 MUAP 峰峰值图特征的高度
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    MultiFeatureDatasetBuilder& setHeight_MUAPP2P(int value);

    /// @brief 设置 MUAP 峰峰值图特征的宽度
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    MultiFeatureDatasetBuilder& setWidth_MUAPP2P(int value);

    /// @brief 设置常见特征的通道数
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    MultiFeatureDatasetBuilder& setNChannels_Comm(int value);

    /// @brief 设置常见特征的深度
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    MultiFeatureDatasetBuilder& setDepth_Comm(int value);

    /// @brief 设置常见特征的高度
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    MultiFeatureDatasetBuilder& setHeight_Comm(int value);

    /// @brief 设置常见特征的宽度
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    MultiFeatureDatasetBuilder& setWidth_Comm(int value);

    /// @brief 创建并返回 MultiFeatureDataset 实例。
    /// @param  
    /// @return 新创建的 MultiFeatureDataset 对象的智能指针。
    std::shared_ptr<MultiFeatureDataset> build(void);

private:
    /** 配置结构体 */
    MultiFeatureDatasetConfig config;

};

