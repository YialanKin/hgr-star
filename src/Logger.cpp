#include "Logger.hpp"

/// @brief 构造日志实例。
/// @param filename 包含文件名的绝对或相对路径，文件类型应为txt。
Logger::Logger(std::string filename) {
    // 日志缓存
    std::vector<std::string> logBuf;

    // 字符串转路径对象
    std::filesystem::path filePath(filename);

    // 相对路径转绝对路径
    if (!filePath.is_absolute()) {
        filePath = std::filesystem::absolute(filePath);
    }

    // 获取日志文件目录 (不包含文件名)
    std::filesystem::path dirPath = filePath.parent_path();

    // 尝试创建日志文件目录
    if (!std::filesystem::exists(dirPath)) {
        if (std::filesystem::create_directories(dirPath)) {
            logBuf.push_back("Log directory \"" + dirPath.string() + "\" created successfully.");
        } else {
            std::cout << "Failed to create log directory." << std::endl;
            exit(100);
        }
    } else {
        logBuf.push_back("Log directory \"" + dirPath.string() + "\" already exists.");
    }

    // 打开文件
    this->m_file = std::ofstream(filename, std::ios_base::app);
    if (this->m_file.is_open()) {
        logBuf.push_back("Object logger constructed successfully.");
    } else {
        std::cout << "Failed to open log file." << std::endl;
        exit(101);
    }

    // 写入日志缓存
    for (const std::string& message : logBuf) {
        this->log(message);
    }
    return;
}

/// @brief 析构日志实例。
/// @param  
Logger::~Logger(void) {
    if (this->m_file.is_open()) {
        this->m_file.close();
    }
    return;
}

/// @brief 输出日志到文件与控制台。
/// @param message 日志消息。
void Logger::log(const std::string& message) {
    std::string outMessage = this->sysTime() + " " + message;
    this->m_file << outMessage << std::endl;    // 输出到日志文件
    std::cout << outMessage << std::endl;       // 输出到控制台
    return;
}

/// @brief 获取当前系统时间。
/// @param  
/// @return 格式化的系统时间字符串。
std::string Logger::sysTime(void) {
    // 获取系统时间
    std::chrono::system_clock::time_point systemTime = std::chrono::system_clock::now();

    // 类型转换
    std::time_t rawSystemTime = std::chrono::system_clock::to_time_t(systemTime);
    
    // 格式化时间
    std::stringstream ss;
    ss << std::put_time(std::localtime(&rawSystemTime), "%c");
    std::string formattedSystemTime = ss.str();

    // 返回
    return formattedSystemTime;
}