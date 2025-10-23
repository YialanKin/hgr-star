#include "StarNet2503132330TVA.hpp"

StarNet2503132330TVA::StarNet2503132330TVA(const StarNet2503132330TVAConfig& config) {
    device = config.device;
    combination = config.combination;

    // 初始化日志记录器
    logger = std::make_shared<Logger>(config.save_path + "\\fold" + std::to_string(config.val_fold) + "\\log.txt");

    // 初始化数据集
    MultiFeatureDatasetBuilder datasetBuilder;
    for (int fold = 1; fold <= 5; ++fold) {
        if (fold == config.val_fold) continue;
        if (config.root_orig != "") datasetBuilder.addPath_Orig(config.root_orig + "\\fold" + std::to_string(fold));
        if (config.root_enve != "") datasetBuilder.addPath_Enve(config.root_enve + "\\fold" + std::to_string(fold));
        if (config.root_muapp2p != "") datasetBuilder.addPath_MUAPP2P(config.root_muapp2p + "\\fold" + std::to_string(fold));
        if (config.root_comm != "") datasetBuilder.addPath_Comm(config.root_comm + "\\fold" + std::to_string(fold));
    }
    std::shared_ptr<MultiFeatureDataset> train_dataset = datasetBuilder.build();
    num_train_batches = (train_dataset->size().value() + config.batch_size - 1) / config.batch_size;
    logger->log("Successfully create train dataset.");
    datasetBuilder.clearPaths();
    if (config.root_orig != "") datasetBuilder.addPath_Orig(config.root_orig + "\\fold" + std::to_string(config.val_fold));
    if (config.root_enve != "") datasetBuilder.addPath_Enve(config.root_enve + "\\fold" + std::to_string(config.val_fold));
    if (config.root_muapp2p != "") datasetBuilder.addPath_MUAPP2P(config.root_muapp2p + "\\fold" + std::to_string(config.val_fold));
    if (config.root_comm != "") datasetBuilder.addPath_Comm(config.root_comm + "\\fold" + std::to_string(config.val_fold));
    std::shared_ptr<MultiFeatureDataset> val_dataset = datasetBuilder.build();
    num_val_batches = (val_dataset->size().value() + config.batch_size - 1) / config.batch_size;
    logger->log("Successfully create val dataset.");

    // 初始化数据加载器
    train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset->map(torch::data::transforms::Stack<>())), torch::data::DataLoaderOptions().batch_size(config.batch_size).workers(8));
    logger->log("Successfully create train data loader.");
    val_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(val_dataset->map(torch::data::transforms::Stack<>())), torch::data::DataLoaderOptions().batch_size(config.batch_size).workers(8));
    logger->log("Successfully create val dataset.");

    // 初始化模型
    torch::autograd::GradMode::set_enabled(true);
    std::shared_ptr<StarNet2503132330Impl> model_ptr = std::make_shared<StarNet2503132330Impl>(config.model_config);
    logger->log("Total number of model parameters: " + std::to_string(Utils::countParameters(model_ptr)));
    model = model_ptr;
    model->to(config.device);
    logger->log("Successfully create model: ");
    for (const auto& named_module : model->named_modules()) {
        logger->log("Layer: " + named_module.key() + " (" + named_module.value()->name() + ")");
    }

    // 初始化优化器
    optimizer = std::make_unique<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions(config.learning_rate).betas({config.beta1, config.beta2}).weight_decay(config.weight_decay));
    logger->log("Successfully create optimizer.");
    
    // 训练循环
    double bestValAcc = 0.0;
    for (int epoch = 0; epoch <= config.epochs; ++epoch) {
        // 学习率调整，热身与余弦退火
        if (epoch <= config.warmup) {
            double learning_rate = (0.9 * epoch / 2 + 0.1) * config.learning_rate;
            for (auto& param_group : optimizer->param_groups()) {
                param_group.options().set_lr(learning_rate);
            }
        } else {
            double learning_rate = 0.5 * (1 + cos(M_PI * (epoch - config.warmup) / (config.epochs - config.warmup))) * config.learning_rate;
            if (learning_rate < config.learning_rate / 10) {
                learning_rate = config.learning_rate / 10;
            }
            for (auto& param_group : optimizer->param_groups()) {
                param_group.options().set_lr(learning_rate);
            }
        }

        // 训练与验证
        std::pair<double, double> train_result = trainOneEpoch(epoch);
        std::pair<double, double> val_result = evaluate(epoch);

        // 保存最优模型
        if (val_result.second > bestValAcc) {
            bestValAcc = val_result.second;
            torch::save(model, config.save_path + "\\fold" + std::to_string(config.val_fold) + "\\bestModel.pt");
        }
    }

    // 保存最后模型
    torch::save(model, config.save_path + "\\fold" + std::to_string(config.val_fold) + "\\lastModel.pt");
    
    logger->log("Best Val Acc: " + std::to_string(bestValAcc));
    return;
}

std::pair<double, double> StarNet2503132330TVA::trainOneEpoch(int epoch) {
    // 计时
    auto epoch_start_time = std::chrono::system_clock::now();               // 该轮次的开始时间
    int64_t batch_duration_sum = 0;                                         // 所有循环的总耗时
    int64_t batch_duration_ave = 0;                                         // 循环的平均用时

    // 训练初始化
    model->train();                                                         // 模型设置为训练模式
    auto loss_function = torch::nn::CrossEntropyLoss();                     // 损失函数选择交叉熵损失
    torch::Tensor accumulated_loss = torch::zeros(1).to(device);            // 累计损失
    torch::Tensor accumulated_goal = torch::zeros(1).to(device);            // 累计正确预测的样本数
    optimizer->zero_grad();                                                 // 优化器归零
    int num_samples = 0;                                                    // 样本数量
    int step = 0;                                                           // 训练步进次数
    float epsilon = 0.1;
    int num_classes = 65;

    // 训练
    for (const auto& batch : *train_loader) {
        auto batch_start_time = std::chrono::system_clock::now();           // 本轮循坏开始时间
        ++step;                                                             // 记录训练步进次数
        torch::Tensor data = batch.data;                                    // 取训练数据
        torch::Tensor x1 = data.narrow(2, 0, 512);
        torch::Tensor x2 = data.narrow(2, 512, 512);
        torch::Tensor x3 = data.narrow(2, 1024, 8);
        torch::Tensor x4 = data.narrow(2, 1032, 4);
        // if (combination == OEM) {
        //     data = torch::concat({x1, x2, x3}, 2);
        // } else if (combination == OEC) {
        //     data = torch::concat({x1, x2, x4}, 2);
        // } else if (combination == OMC) {
        //     data = torch::concat({x1, x3, x4}, 2);
        // } else if (combination == EMC) {
        //     data = torch::concat({x2, x3, x4}, 2);
        // } else {
        //     exit(2503030858);
        // }
        data = x1;
        torch::Tensor classes = batch.target;                               // 取训练标签
        torch::Tensor onehot = torch::zeros({classes.size(0), num_classes});
        onehot.scatter_(1, classes.unsqueeze(1).to(torch::kLong), 1);
        torch::Tensor smoothed_part = torch::ones_like(onehot) * (epsilon / (num_classes - 1));
        torch::Tensor one_minus_epsilon = torch::ones_like(onehot) * (1 - epsilon);
        torch::Tensor smoothed_labels = torch::where(onehot == 1, one_minus_epsilon, smoothed_part);
        num_samples += data.size(0);                                        // 累计样本数量
        torch::Tensor pred = model->forward(data.to(torch::kF32).to(device));
        torch::Tensor pred_classes = pred.argmax(1);                        // 预测标签
        accumulated_goal += pred_classes.eq(classes.to(device)).sum();      // 累计正确预测的样本数
        torch::Tensor loss = loss_function(pred, smoothed_labels.to(device));       // 计算损失
        loss.backward();                                                    // 反向传播
        accumulated_loss += loss.item<float>();                             // 累计损失
        optimizer->step();                                                  // 优化器步进
        optimizer->zero_grad();                                             // 优化器归零
        auto batch_end_time = std::chrono::system_clock::now();             // 本轮循坏结束时间
        int64_t batch_duration = computeDuration_ms(batch_start_time, batch_end_time);                      // 计算该轮循环的耗时
        batch_duration_sum += batch_duration;                               // 累计循环用时
        batch_duration_ave = batch_duration_sum / step;                     // 计算平均循环用时
        std::vector<int64_t> epoch_duration_hms = computeDuration_hms(epoch_start_time, batch_end_time);    // 计算该轮次的耗时
        int64_t epoch_duration_ms = this->computeDuration_ms(epoch_start_time, batch_end_time);             // 计算该轮次的耗时
        int64_t num_batch_remain = num_train_batches - step;                // 剩余批次数
        int64_t remain_time_s = num_batch_remain * (epoch_duration_ms / step) / 1000;                       // 预计训练结束用时
        std::stringstream log_stream;                                        // 日志流
        log_stream << " Batch time used: ";
        log_stream << std::setw(3) << std::setfill(' ') << batch_duration << "ms (average: " << std::setw(3) << std::setfill(' ') << batch_duration_ave << "ms). ";
        log_stream << "Epoch time used: " << std::setw(2) << std::setfill('0') << epoch_duration_hms[0] << ":" << std::setw(2) << std::setfill('0') << epoch_duration_hms[1] << ":" << std::setw(2) << std::setfill('0') << epoch_duration_hms[2] << ". ";
        log_stream << "Estimated remain time: " << std::setw(2) << std::setfill('0') << remain_time_s / 3600 << ":" << std::setw(2) << std::setfill('0') << (remain_time_s % 3600) / 60 << ":" << std::setw(2) << std::setfill('0') << remain_time_s % 60 << ". ";
        log_stream << "    Train epoch: " << std::setw(2) << std::setfill(' ') << epoch << " | ";
        log_stream << "Batch: " << std::setw(3) << std::setfill(' ') << step << " | ";
        log_stream << "Loss: " << std::fixed << std::setprecision(6) << accumulated_loss.item<float>() / step << " | ";
        log_stream << "Acc: " << std::fixed << std::setprecision(6) << accumulated_goal.item<float>() / num_samples << " ";
        logger->log(log_stream.str());
    }
    return std::make_pair(accumulated_loss.item<float>() / step, accumulated_goal.item<float>() / num_samples);
}

std::pair<double, double> StarNet2503132330TVA::evaluate(int epoch) {
    // 计时
    auto epoch_start_time = std::chrono::system_clock::now();               // 该轮次的开始时间
    int64_t batch_duration_sum = 0;                                         // 所有循环的总耗时
    int64_t batch_duration_ave = 0;                                         // 循环的平均用时

    // 验证初始化
    model->eval();                                                          // 模型设置为验证模式
    auto loss_function = torch::nn::CrossEntropyLoss();                     // 损失函数选择交叉熵损失
    torch::Tensor accumulated_loss = torch::zeros(1).to(device);            // 累计损失
    torch::Tensor accumulated_goal = torch::zeros(1).to(device);            // 累计正确预测的样本数
    int64_t num_samples = 0;                                                // 样本数量
    int64_t step = 0;                                                       // 验证步进次数

    // 验证
    for (const auto& batch : *val_loader) {
        auto batch_start_time = std::chrono::system_clock::now();           // 本轮循坏开始时间
        ++step;                                                             // 记录验证步进次数
        torch::Tensor data = batch.data;                                    // 取验证数据
        torch::Tensor x1 = data.narrow(2, 0, 512);
        torch::Tensor x2 = data.narrow(2, 512, 512);
        torch::Tensor x3 = data.narrow(2, 1024, 8);
        torch::Tensor x4 = data.narrow(2, 1032, 4);
        // if (combination == OEM) {
        //     data = torch::concat({x1, x2, x3}, 2);
        // } else if (combination == OEC) {
        //     data = torch::concat({x1, x2, x4}, 2);
        // } else if (combination == OMC) {
        //     data = torch::concat({x1, x3, x4}, 2);
        // } else if (combination == EMC) {
        //     data = torch::concat({x2, x3, x4}, 2);
        // } else {
        //     exit(2503030859);
        // }
        data = x1;
        torch::Tensor classes = batch.target;                               // 取验证标签
        num_samples += data.size(0);                                        // 累计样本数量
        torch::Tensor pred = model->forward(data.to(torch::kF32).to(device));
        torch::Tensor pred_classes = pred.argmax(1);                        // 预测标签
        accumulated_goal += pred_classes.eq(classes.to(device)).sum();      // 累计正确预测的样本数
        torch::Tensor loss = loss_function(pred, classes.to(device));       // 计算损失
        accumulated_loss += loss.item<float>();                             // 累计损失
        auto batch_end_time = std::chrono::system_clock::now();             // 本轮循坏结束时间
        int64_t batch_duration = computeDuration_ms(batch_start_time, batch_end_time);                          // 计算该轮循环的耗时
        batch_duration_sum += batch_duration;                               // 累计循环用时
        batch_duration_ave = batch_duration_sum / step;                     // 计算平均循环用时
        std::vector<int64_t> epoch_duration_hms = computeDuration_hms(epoch_start_time, batch_end_time);        // 计算该轮次的耗时
        int64_t epoch_duration_ms = computeDuration_ms(epoch_start_time, batch_end_time);                       // 计算该轮次的耗时
        int64_t num_batch_remain = num_val_batches - step;                  // 剩余批次数
        int64_t remain_time_s = num_batch_remain * (epoch_duration_ms / step) / 1000;                           // 预计验证结束用时
        std::stringstream log_stream;                                        // 日志流
        log_stream << " Batch time used: ";
        log_stream << std::setw(3) << std::setfill(' ') << batch_duration << "ms (average: " << std::setw(3) << std::setfill(' ') << batch_duration_ave << "ms). ";
        log_stream << "Epoch time used: " << std::setw(2) << std::setfill('0') << epoch_duration_hms[0] << ":" << std::setw(2) << std::setfill('0') << epoch_duration_hms[1] << ":" << std::setw(2) << std::setfill('0') << epoch_duration_hms[2] << ". ";
        log_stream << "Estimated remain time: " << std::setw(2) << std::setfill('0') << remain_time_s / 3600 << ":" << std::setw(2) << std::setfill('0') << (remain_time_s % 3600) / 60 << ":" << std::setw(2) << std::setfill('0') << remain_time_s % 60 << ". ";
        log_stream << "    Val epoch: " << std::setw(2) << std::setfill(' ') << epoch << " | ";
        log_stream << "Batch: " << std::setw(3) << std::setfill(' ') << step << " | ";
        log_stream << "Loss: " << std::fixed << std::setprecision(6) << accumulated_loss.item<float>() / step << " | ";
        log_stream << "Acc: " << std::fixed << std::setprecision(6) << accumulated_goal.item<float>() / num_samples << " ";
        logger->log(log_stream.str());
    }

    // 计时
    auto epochEndTime = std::chrono::system_clock::now();        // 该轮次的结束时间

    return std::make_pair(accumulated_loss.item<float>() / step, accumulated_goal.item<float>() / num_samples);
}

int64_t StarNet2503132330TVA::computeDuration_ms(const std::chrono::system_clock::time_point& start, const std::chrono::system_clock::time_point& end) {
    auto duration = end - start;
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    return milliseconds;
}

std::vector<int64_t> StarNet2503132330TVA::computeDuration_hms(const std::chrono::system_clock::time_point& start, const std::chrono::system_clock::time_point& end) {
    auto duration = end - start;
    auto total_seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
    int64_t hours = total_seconds / 3600;
    int64_t minutes = (total_seconds % 3600) / 60;
    int64_t seconds = total_seconds % 60;
    return std::vector<int64_t>{hours, minutes, seconds};
}



StarNet2503132330TVABuilder& StarNet2503132330TVABuilder::setCUDA(bool value) {
    if (!value) return *this;
    if (!torch::cuda::is_available()) return *this;
    config.device = torch::Device(torch::kCUDA);
    return *this;
}

StarNet2503132330TVABuilder& StarNet2503132330TVABuilder::setEpochs(int value) {
    config.epochs = value;
    return *this;
}

StarNet2503132330TVABuilder& StarNet2503132330TVABuilder::setWarmup(int value) {
    config.warmup = value;
    return *this;
}

StarNet2503132330TVABuilder& StarNet2503132330TVABuilder::setBatchSize(int value) {
    config.batch_size = value;
    return *this;
}

StarNet2503132330TVABuilder& StarNet2503132330TVABuilder::setLearningRate(double value) {
    config.learning_rate = value;
    return *this;
}

StarNet2503132330TVABuilder& StarNet2503132330TVABuilder::setBeta1(double value) {
    config.beta1 = value;
    return *this;
}

StarNet2503132330TVABuilder& StarNet2503132330TVABuilder::setBeta2(double value) {
    config.beta2 = value;
    return *this;
}

StarNet2503132330TVABuilder& StarNet2503132330TVABuilder::setWeightDecay(double value) {
    config.weight_decay = value;
    return *this;
}

StarNet2503132330TVABuilder& StarNet2503132330TVABuilder::setRootOrig(std::string value) {
    config.root_orig = value;
    return *this;
}

StarNet2503132330TVABuilder& StarNet2503132330TVABuilder::setRootEnve(std::string value) {
    config.root_enve = value;
    return *this;
}

StarNet2503132330TVABuilder& StarNet2503132330TVABuilder::setRootMUAPP2P(std::string value) {
    config.root_muapp2p = value;
    return *this;
}

StarNet2503132330TVABuilder& StarNet2503132330TVABuilder::setRootComm(std::string value) {
    config.root_comm = value;
    return *this;
}

StarNet2503132330TVABuilder& StarNet2503132330TVABuilder::setSavePath(std::string value) {
    config.save_path = value;
    return *this;
}

StarNet2503132330TVABuilder& StarNet2503132330TVABuilder::setValFold(int value) {
    config.val_fold = value;
    return *this;
}

StarNet2503132330TVABuilder& StarNet2503132330TVABuilder::setCombination(int value) {
    config.combination = value;
    return *this;
}

std::shared_ptr<StarNet2503132330TVA> StarNet2503132330TVABuilder::build(void) {
    return std::make_shared<StarNet2503132330TVA>(config);
}





