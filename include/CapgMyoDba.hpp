#pragma once
#include <torch/torch.h>
#include <filesystem>
#include <fstream>

struct CapgMyoDbaConfig {
    /** 原始特征五折目录列表 */
    std::vector<std::string> paths_orig;

    /** 原始特征的通道数 */
    int nchannels_orig = 1;

    /** 原始特征的深度 */
    int depth_orig = 200;
    
    /** 原始特征的高度 */
    int height_orig = 8;

    /** 原始特征的宽度 */
    int width_orig = 16;

    /** 包络特征五折目录列表 */
    std::vector<std::string> paths_enve;

    /** 包络特征的通道数 */
    int nchannels_enve = 1;

    /** 包络特征的深度 */
    int depth_enve = 200;
    
    /** 包络特征的高度 */
    int height_enve = 8;

    /** 包络特征的宽度 */
    int width_enve = 16;

    /** MUAP 峰峰值图特征五折目录列表 */
    std::vector<std::string> paths_muapp2p;

    /** MUAP 峰峰值图特征的通道数 */
    int nchannels_muapp2p = 1;

    /** MUAP 峰峰值图特征的深度 */
    int depth_muapp2p = 8;
    
    /** MUAP 峰峰值图特征的高度 */
    int height_muapp2p = 8;

    /** MUAP 峰峰值图特征的宽度 */
    int width_muapp2p = 16;

    /** 常见特征 (MAV, ZCR, WL, Willison Amplitude) 五折目录列表 */
    std::vector<std::string> paths_comm;

    /** 常见特征的通道数 */
    int nchannels_comm = 1;

    /** 常见特征的深度 */
    int depth_comm = 8;
    
    /** 常见特征的高度 */
    int height_comm = 8;

    /** 常见特征的宽度 */
    int width_comm = 16;

};

class CapgMyoDba : public torch::data::Dataset<CapgMyoDba> {
public:
    /// @brief CapgMyoDba 的构造函数，接受配置结构体并根据其中的参数完成数据集的初始化。
    /// @param config CapgMyoDba 的配置结构体。
    CapgMyoDba(const CapgMyoDbaConfig& config);

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

class CapgMyoDbaBuilder {
public:
    /// @brief 添加一个原始特征的折目录。
    /// @param value 折目录。
    /// @return 当前构建器对象，支持链式调用。
    CapgMyoDbaBuilder& addPath_Orig(std::string value);

    /// @brief 添加一个包络特征的折目录。
    /// @param value 折目录。
    /// @return 当前构建器对象，支持链式调用。
    CapgMyoDbaBuilder& addPath_Enve(std::string value);

    /// @brief 添加一个 MUAP 峰峰值特征的折目录。
    /// @param value 折目录。
    /// @return 当前构建器对象，支持链式调用。
    CapgMyoDbaBuilder& addPath_MUAPP2P(std::string value);

    /// @brief 添加一个常见特征的折目录。
    /// @param value 折目录。
    /// @return 当前构建器对象，支持链式调用。
    CapgMyoDbaBuilder& addPath_Comm(std::string value);

    /// @brief 清空折目录。
    /// @param 
    /// @return 当前构建器对象，支持链式调用。
    CapgMyoDbaBuilder& clearPaths(void);

    /// @brief 设置原始特征的通道数
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    CapgMyoDbaBuilder& setNChannels_Orig(int value);

    /// @brief 设置原始特征的深度
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    CapgMyoDbaBuilder& setDepth_Orig(int value);

    /// @brief 设置原始特征的高度
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    CapgMyoDbaBuilder& setHeight_Orig(int value);

    /// @brief 设置原始特征的宽度
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    CapgMyoDbaBuilder& setWidth_Orig(int value);

    /// @brief 设置包络特征的通道数
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    CapgMyoDbaBuilder& setNChannels_Enve(int value);

    /// @brief 设置包络特征的深度
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    CapgMyoDbaBuilder& setDepth_Enve(int value);

    /// @brief 设置包络特征的高度
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    CapgMyoDbaBuilder& setHeight_Enve(int value);

    /// @brief 设置包络特征的宽度
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    CapgMyoDbaBuilder& setWidth_Enve(int value);

    /// @brief 设置 MUAP 峰峰值图特征的通道数
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    CapgMyoDbaBuilder& setNChannels_MUAPP2P(int value);

    /// @brief 设置 MUAP 峰峰值图特征的深度
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    CapgMyoDbaBuilder& setDepth_MUAPP2P(int value);

    /// @brief 设置 MUAP 峰峰值图特征的高度
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    CapgMyoDbaBuilder& setHeight_MUAPP2P(int value);

    /// @brief 设置 MUAP 峰峰值图特征的宽度
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    CapgMyoDbaBuilder& setWidth_MUAPP2P(int value);

    /// @brief 设置常见特征的通道数
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    CapgMyoDbaBuilder& setNChannels_Comm(int value);

    /// @brief 设置常见特征的深度
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    CapgMyoDbaBuilder& setDepth_Comm(int value);

    /// @brief 设置常见特征的高度
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    CapgMyoDbaBuilder& setHeight_Comm(int value);

    /// @brief 设置常见特征的宽度
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    CapgMyoDbaBuilder& setWidth_Comm(int value);

    /// @brief 创建并返回 CapgMyoDba 实例。
    /// @param  
    /// @return 新创建的 CapgMyoDba 对象的智能指针。
    std::shared_ptr<CapgMyoDba> build(void);

private:
    /** 配置结构体 */
    CapgMyoDbaConfig config;

};

