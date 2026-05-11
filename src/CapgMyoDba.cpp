/**
 * @file CapgMyoDba.cpp
 * @brief 包含 CapgMyoDba 类、配置结构体及构建器类的实现。
 * @author Rat Kin
 * @date 2025-03-02
 * 
 * @note 2025-03-06 使用torch::empty代替torch::zeros, 期望提升效率
 */

#include "CapgMyoDba.hpp"

CapgMyoDba::CapgMyoDba(const CapgMyoDbaConfig& config) {
    // 初始化原始特征的文件列表
    files_orig = std::vector<std::string>();
    for (const std::string& path : config.paths_orig) {
        // std::cout << "1: " << path << std::endl;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
            if (!std::filesystem::is_regular_file(entry.status())) continue;
            files_orig.push_back(entry.path().string());
        }
    }
    sort(files_orig.begin(), files_orig.end());
    nchannels_orig = config.nchannels_orig;
    depth_orig = config.depth_orig;
    height_orig = config.height_orig;
    width_orig = config.width_orig;
    datalen_orig = nchannels_orig * depth_orig * height_orig * width_orig;
    // std::cout << files_orig.size() << std::endl;

    // 初始化包络特征的文件列表
    files_enve = std::vector<std::string>();
    for (const std::string& path : config.paths_enve) {
        // std::cout << "2: " << path << std::endl;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
            if (!std::filesystem::is_regular_file(entry.status())) continue;
            files_enve.push_back(entry.path().string());
        }
    }
    sort(files_enve.begin(), files_enve.end());
    nchannels_enve = config.nchannels_enve;
    depth_enve = config.depth_enve;
    height_enve = config.height_enve;
    width_enve = config.width_enve;
    datalen_enve = nchannels_enve * depth_enve * height_enve * width_enve;
    // std::cout << files_enve.size() << std::endl;

    // 初始化 MUAP 峰峰值特征的文件列表
    files_muapp2p = std::vector<std::string>();
    for (const std::string& path : config.paths_muapp2p) {
        // std::cout << "3: " << path << std::endl;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
            if (!std::filesystem::is_regular_file(entry.status())) continue;
            files_muapp2p.push_back(entry.path().string());
        }
    }
    sort(files_muapp2p.begin(), files_muapp2p.end());
    nchannels_muapp2p = config.nchannels_muapp2p;
    depth_muapp2p = config.depth_muapp2p;
    height_muapp2p = config.height_muapp2p;
    width_muapp2p = config.width_muapp2p;
    datalen_muapp2p = nchannels_muapp2p * depth_muapp2p * height_muapp2p * width_muapp2p;
    // std::cout << files_muapp2p.size() << std::endl;

    // 初始化常见特征的文件列表
    files_comm = std::vector<std::string>();
    for (const std::string& path : config.paths_comm) {
        // std::cout << "4: " << path << std::endl;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
            if (!std::filesystem::is_regular_file(entry.status())) continue;
            files_comm.push_back(entry.path().string());
        }
    }
    sort(files_comm.begin(), files_comm.end());
    nchannels_comm = config.nchannels_comm;
    depth_comm = config.depth_comm;
    height_comm = config.height_comm;
    width_comm = config.width_comm;
    datalen_comm = nchannels_comm * depth_comm * height_comm * width_comm;
    // std::cout << files_comm.size() << std::endl;

    // 初始化样本标签列表
    std::vector<std::string> notempty;
    if (!files_orig.empty()) {
        notempty = files_orig;
    } else if (!files_enve.empty()) {
        notempty = files_enve;
    } else if (!files_muapp2p.empty()) {
        notempty = files_muapp2p;
    } else {
        notempty = files_comm;
    }
    classes = std::vector<int>(notempty.size());
    for (size_t i = 0; i < notempty.size(); ++i) {
        std::string filename = notempty[i];

        // 提取样本路径中的类别子串 "class_X"
        std::string class_str(filename.begin() + filename.find("class"), filename.begin() + filename.find("\\sample"));

        // 提取类别子串中的类别值 "X"
        std::string class_value_str(class_str.begin() + class_str.find("_") + 1, class_str.end());

        // 记录标签
        classes[i] = std::stoi(class_value_str);
    }

    return;
}

torch::optional<size_t> CapgMyoDba::size() const {
    if (!files_orig.empty()) return files_orig.size();
    if (!files_enve.empty()) return files_enve.size();
    if (!files_muapp2p.empty()) return files_muapp2p.size();
    return files_comm.size();
}

torch::data::Example<> CapgMyoDba::get(size_t index) {
    // 读取原始特征的二进制文件
    torch::Tensor tensor_orig;
    if (files_orig.empty()) {
        tensor_orig = torch::empty({nchannels_orig, depth_orig, height_orig, width_orig}, torch::kF32);
    } else {
        std::string filename = files_orig[index];
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "Failed to open file \"" + filename + "\"" << std::endl;
            exit(2503020903);
        }
        std::vector<float> data_orig(datalen_orig);
        file.read(reinterpret_cast<char*>(data_orig.data()), sizeof(float) * datalen_orig);
        file.close();
        tensor_orig = torch::from_blob(data_orig.data(), {nchannels_orig, depth_orig, height_orig, width_orig}, torch::kF32).clone();
    }

    // 读取包络特征的二进制文件
    torch::Tensor tensor_enve;
    if (files_enve.empty()) {
        tensor_enve = torch::empty({nchannels_enve, depth_enve, height_enve, width_enve}, torch::kF32);
    } else {
        std::string filename = files_enve[index];
        std::ifstream file = std::ifstream(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "Failed to open file \"" + filename + "\"" << std::endl;
            exit(2503020905);
        }
        std::vector<float> data_enve(datalen_enve);
        file.read(reinterpret_cast<char*>(data_enve.data()), sizeof(float) * datalen_enve);
        file.close();
        tensor_enve = torch::from_blob(data_enve.data(), {nchannels_enve, depth_enve, height_enve, width_enve}, torch::kF32).clone();
    }

    // 读取 MUAP 峰峰值图特征的二进制文件
    torch::Tensor tensor_muapp2p;
    if (files_muapp2p.empty()) {
        tensor_muapp2p = torch::empty({nchannels_muapp2p, depth_muapp2p, height_muapp2p, width_muapp2p}, torch::kF32);
    } else {
        std::string filename = files_muapp2p[index];
        std::ifstream file = std::ifstream(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "Failed to open file \"" + filename + "\"" << std::endl;
            exit(2503020907);
        }
        std::vector<float> data_muapp2p(datalen_muapp2p);
        file.read(reinterpret_cast<char*>(data_muapp2p.data()), sizeof(float) * datalen_muapp2p);
        file.close();
        tensor_muapp2p = torch::from_blob(data_muapp2p.data(), {nchannels_muapp2p, depth_muapp2p, height_muapp2p, width_muapp2p}, torch::kF32).clone();
    }


    // 读取常见特征的二进制文件
    torch::Tensor tensor_comm;
    if (files_comm.empty()) {
        tensor_comm = torch::empty({nchannels_comm, depth_comm, height_comm, width_comm}, torch::kF32);
    } else {
        std::string filename = files_comm[index];
        std::ifstream file = std::ifstream(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "Failed to open file \"" + filename + "\"" << std::endl;
            exit(2503020908);
        }
        std::vector<float> data_comm(datalen_comm);
        file.read(reinterpret_cast<char*>(data_comm.data()), sizeof(float) * datalen_comm);
        file.close();
        tensor_comm = torch::from_blob(data_comm.data(), {nchannels_comm, depth_comm, height_comm, width_comm}, torch::kF32).clone();
    }

    // 获取标签
    torch::Tensor sample_class = torch::tensor(classes[index] - 1);

    // 输出
    return {torch::cat({tensor_orig, tensor_enve, tensor_muapp2p, tensor_comm}, 1), sample_class};
}

CapgMyoDbaBuilder& CapgMyoDbaBuilder::addPath_Orig(std::string value) {
    config.paths_orig.push_back(value);
    return *this;
}

CapgMyoDbaBuilder& CapgMyoDbaBuilder::addPath_Enve(std::string value) {
    config.paths_enve.push_back(value);
    return *this;
}

CapgMyoDbaBuilder& CapgMyoDbaBuilder::addPath_MUAPP2P(std::string value) {
    config.paths_muapp2p.push_back(value);
    return *this;
}

CapgMyoDbaBuilder& CapgMyoDbaBuilder::addPath_Comm(std::string value) {
    config.paths_comm.push_back(value);
    return *this;
}

CapgMyoDbaBuilder& CapgMyoDbaBuilder::clearPaths(void) {
    config.paths_orig.clear();
    config.paths_enve.clear();
    config.paths_muapp2p.clear();
    config.paths_comm.clear();
    return *this;
}

CapgMyoDbaBuilder& CapgMyoDbaBuilder::setNChannels_Orig(int value) {
    config.nchannels_orig = value;
    return *this;
}

CapgMyoDbaBuilder& CapgMyoDbaBuilder::setDepth_Orig(int value) {
    config.depth_orig = value;
    return *this;
}

CapgMyoDbaBuilder& CapgMyoDbaBuilder::setHeight_Orig(int value) {
    config.height_orig = value;
    return *this;
}

CapgMyoDbaBuilder& CapgMyoDbaBuilder::setWidth_Orig(int value) {
    config.width_orig = value;
    return *this;
}

CapgMyoDbaBuilder& CapgMyoDbaBuilder::setNChannels_Enve(int value) {
    config.nchannels_enve = value;
    return *this;
}

CapgMyoDbaBuilder& CapgMyoDbaBuilder::setDepth_Enve(int value) {
    config.depth_enve = value;
    return *this;
}

CapgMyoDbaBuilder& CapgMyoDbaBuilder::setHeight_Enve(int value) {
    config.height_enve = value;
    return *this;
}

CapgMyoDbaBuilder& CapgMyoDbaBuilder::setWidth_Enve(int value) {
    config.width_enve = value;
    return *this;
}

CapgMyoDbaBuilder& CapgMyoDbaBuilder::setNChannels_MUAPP2P(int value) {
    config.nchannels_muapp2p = value;
    return *this;
}

CapgMyoDbaBuilder& CapgMyoDbaBuilder::setDepth_MUAPP2P(int value) {
    config.depth_muapp2p = value;
    return *this;
}

CapgMyoDbaBuilder& CapgMyoDbaBuilder::setHeight_MUAPP2P(int value) {
    config.height_muapp2p = value;
    return *this;
}

CapgMyoDbaBuilder& CapgMyoDbaBuilder::setWidth_MUAPP2P(int value) {
    config.width_muapp2p = value;
    return *this;
}

CapgMyoDbaBuilder& CapgMyoDbaBuilder::setNChannels_Comm(int value) {
    config.nchannels_comm = value;
    return *this;
}

CapgMyoDbaBuilder& CapgMyoDbaBuilder::setDepth_Comm(int value) {
    config.depth_comm = value;
    return *this;
}

CapgMyoDbaBuilder& CapgMyoDbaBuilder::setHeight_Comm(int value) {
    config.height_comm = value;
    return *this;
}

CapgMyoDbaBuilder& CapgMyoDbaBuilder::setWidth_Comm(int value) {
    config.width_comm = value;
    return *this;
}

std::shared_ptr<CapgMyoDba> CapgMyoDbaBuilder::build(void) {
    return std::make_shared<CapgMyoDba>(config);
}