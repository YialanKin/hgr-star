#pragma once
#include "StarNet2606062000.hpp"
#include "CapgMyoDba.hpp"
#include "Logger.hpp"
#include "Utils.hpp"

#define OEM 1
#define OEC 2
#define OMC 3
#define EMC 4

struct StarNet2606062000TVConfig {
    /** 模型运行设备 */
    torch::Device device = torch::kCPU;

    /** 训练时期数 */
    int epochs = 0;

    /** 热身时期数 */
    int warmup = 0;

    /** 批次大小 */
    int batch_size = 0;

    /** Adam 优化器的学习率 */
    double learning_rate = 0.0;

    /** Adam 优化器的 beta1 */
    double beta1 = 0.9;

    /** Adam 优化器的 beta2 */
    double beta2 = 0.999;

    /** Adam 优化器的 weight decay */
    double weight_decay = 0.001;

    /** 原始特征目录 */
    std::string root_orig = "";

    /** 包络特征目录 */
    std::string root_enve = "";

    /** MUAP 峰峰值图特征目录 */
    std::string root_muapp2p = "";

    /** 常见特征目录 */
    std::string root_comm = "";

    /** 保存目录 */
    std::string save_path = "";

    /** 验证折 */
    int val_fold = 0;

    /** 特征组合方式 */
    int combination = -1;

    /** 模型的配置结构体 */
    StarNet2606062000Options model_config;
};


class StarNet2606062000TV {
public:
    /// @brief StarNet2606062000TV 的构造函数
    /// @param config StarNet2606062000TV 的配置结构体
    StarNet2606062000TV(const StarNet2606062000TVConfig& config);

private:
    /// @brief 训练一个时期
    /// @param epoch 时期
    /// @return 该时期的累计损失与累计正确预测数
    std::pair<double, double> trainOneEpoch(int epoch);

    /// @brief 验证一个时期
    /// @param epoch 时期
    /// @return 该时期的累计损失与累计正确预测数
    std::pair<double, double> evaluate(int epoch);

    /// @brief 计算两个时间点的毫秒间隔
    /// @param start 开始时间点
    /// @param end 结束时间点
    /// @return 两个时间点的毫秒间隔
    int64_t computeDuration_ms(const std::chrono::system_clock::time_point& start, const std::chrono::system_clock::time_point& end);
    
    /// @brief 计算两个时间点的标准时间隔
    /// @param start 开始时间点
    /// @param end 结束时间点
    /// @return 两个时间点的标准时间隔
    std::vector<int64_t> computeDuration_hms(const std::chrono::system_clock::time_point& start, const std::chrono::system_clock::time_point& end);


    /** 日志记录器 */
    std::shared_ptr<Logger> logger = nullptr;

    /** 训练批次数量 */
    int num_train_batches;

    /** 验证批次数量 */
    int num_val_batches;

    /** 训练集加载器 */
    std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<CapgMyoDba, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::RandomSampler>, std::default_delete<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<CapgMyoDba, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::RandomSampler> > >
        train_loader = nullptr;

    /** 验证集加载器 */
    std::unique_ptr<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<CapgMyoDba, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::RandomSampler>, std::default_delete<torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<CapgMyoDba, torch::data::transforms::Stack<torch::data::Example<at::Tensor, at::Tensor> > >, torch::data::samplers::RandomSampler> > >
        val_loader = nullptr;

    /** 模型 */
    StarNet2606062000 model = nullptr;

    /** 优化器 */
    std::unique_ptr<torch::optim::Adam> optimizer = nullptr;

    /** 模型运行设备 */
    torch::Device device = torch::kCPU;

    /** 特征组合 */
    int combination;
};

class StarNet2606062000TVBuilder {
public:
    /// @brief 设置模型是否运行于 GPU
    /// @param value 运行于 GPU 的标志
    /// @return 当前构建器对象，支持链式调用
    StarNet2606062000TVBuilder& setCUDA(bool value);

    /// @brief 设置训练时期数。
    /// @param value 训练时期数。
    /// @return 当前构建器对象，支持链式调用。
    StarNet2606062000TVBuilder& setEpochs(int value);

    /// @brief 设置热身时期数。
    /// @param value 热身时期数。
    /// @return 当前构建器对象，支持链式调用。
    StarNet2606062000TVBuilder& setWarmup(int value);

    /// @brief 设置批次大小。
    /// @param value 批次大小。
    /// @return 当前构建器对象，支持链式调用。
    StarNet2606062000TVBuilder& setBatchSize(int value);

    /// @brief 设置学习率。
    /// @param value 学习率。
    /// @return 当前构建器对象，支持链式调用。
    StarNet2606062000TVBuilder& setLearningRate(double value);

    /// @brief 设置 Adam 优化器的 beta1。
    /// @param value Adam 优化器的 beta1。
    /// @return 当前构建器对象，支持链式调用。
    StarNet2606062000TVBuilder& setBeta1(double value);

    /// @brief 设置 Adam 优化器的 beta2。
    /// @param value Adam 优化器的 beta2。
    /// @return 当前构建器对象，支持链式调用。
    StarNet2606062000TVBuilder& setBeta2(double value);

    /// @brief 设置 Adam 优化器的 weight decay。
    /// @param value Adam 优化器的 weight decay。
    /// @return 当前构建器对象，支持链式调用。
    StarNet2606062000TVBuilder& setWeightDecay(double value);

    /// @brief 设置原始特征目录
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    StarNet2606062000TVBuilder& setRootOrig(std::string value);

    /// @brief 设置包络特征目录
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    StarNet2606062000TVBuilder& setRootEnve(std::string value);

    /// @brief 设置 MUAP 峰峰值图特征目录
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    StarNet2606062000TVBuilder& setRootMUAPP2P(std::string value);

    /// @brief 设置常见特征目录
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    StarNet2606062000TVBuilder& setRootComm(std::string value);

    /// @brief 设置保存目录。
    /// @param value 保存目录。
    /// @return 当前构建器对象，支持链式调用。
    StarNet2606062000TVBuilder& setSavePath(std::string value);

    /// @brief 设置验证折。
    /// @param value 验证折。
    /// @return 当前构建器对象，支持链式调用。
    StarNet2606062000TVBuilder& setValFold(int value);

    /// @brief 设置特征组合
    /// @param value 设定值
    /// @return 当前构建器对象，支持链式调用
    StarNet2606062000TVBuilder& setCombination(int value);

    /// @brief 创建并返回 StarNet2606062000TV 实例。
    /// @param  
    /// @return 新创建的 StarNet2606062000TV 对象的智能指针。
    std::shared_ptr<StarNet2606062000TV> build(void);
    
    /** 训练程序的配置结构体 */
    StarNet2606062000TVConfig config;

};

