#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <chrono>

class Logger {
private:
    std::ofstream m_file;

public:
    Logger(void) = default;
    Logger(std::string filename);
    ~Logger(void);

    void log(const std::string& message);
    std::string sysTime(void);
};
