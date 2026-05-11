#include "Utils.hpp"

void Utils::summarize(std::string res_save, std::string sum_save) {
    std::vector<std::vector<double>> BestRes(21, std::vector<double>(6, 0.0));
    std::vector<std::vector<double>> BestResMacroP(21, std::vector<double>(6, 0.0));
    std::vector<std::vector<double>> BestResMacroR(21, std::vector<double>(6, 0.0));
    std::vector<std::vector<double>> BestResMacroF1(21, std::vector<double>(6, 0.0));
    std::vector<std::vector<double>> LastRes(21, std::vector<double>(6, 0.0));
    for (int s = 1; s <= 20; ++s) {
        for (int fold = 1; fold <= 5; ++fold) {
            std::vector<std::string> text;
            // Read file
            std::string file = res_save + "\\s" + std::to_string(s) + "\\fold" + std::to_string(fold) + "\\log.txt";
            std::ifstream infile(file);
            std::string line;
            if (!infile.is_open()) {
                std::cerr << "Unable to open file:" << file << std::endl;
                continue;
            }
            while (getline(infile, line)) {
                text.push_back(line);
            }
            infile.close();

            bool accfind = false;
            bool mpfind = false;
            bool mrfind = false;
            bool mf1find = false;
            for (int i = text.size() - 1; i >= 0 && (!accfind || !mpfind || !mrfind || !mf1find); --i) {
                std::string line = text[i];
                std::string acc_str = "Best Val Acc: ";
                int acc_find_res = line.find(acc_str);
                if (acc_find_res != std::string::npos) {
                    // std::cout << std::string(line.begin() + acc_find_res + acc_str.size(), line.end()) << std::endl;
                    BestRes[s][fold] = std::stod(std::string(line.begin() + acc_find_res + acc_str.size(), line.end()));
                    accfind = true;
                    continue;
                }

                std::string mp_str = "Best Val Macro Precision: ";
                int mp_find_res = line.find(mp_str);
                if (mp_find_res != std::string::npos) {
                    // std::cout << std::string(line.begin() + mp_find_res + mp_str.size(), line.end()) << std::endl;
                    BestResMacroP[s][fold] = std::stod(std::string(line.begin() + mp_find_res + mp_str.size(), line.end()));
                    mpfind = true;
                    continue;
                }

                std::string mr_str = "Best Val Macro Recall: ";
                int mr_find_res = line.find(mr_str);
                if (mr_find_res != std::string::npos) {
                    // std::cout << std::string(line.begin() + mr_find_res + mr_str.size(), line.end()) << std::endl;
                    BestResMacroR[s][fold] = std::stod(std::string(line.begin() + mr_find_res + mr_str.size(), line.end()));
                    mrfind = true;
                    continue;
                }
                
                std::string mf1_str = "Best Val Macro F1: ";
                int mf1_find_res = line.find(mf1_str);
                if (mf1_find_res != std::string::npos) {
                    // std::cout << std::string(line.begin() + mf1_find_res + mf1_str.size(), line.end()) << std::endl;
                    BestResMacroF1[s][fold] = std::stod(std::string(line.begin() + mf1_find_res + mf1_str.size(), line.end()));
                    mf1find = true;
                    continue;
                }
            }
            if (accfind && mpfind && mrfind && mf1find) {
                continue;
            }

            // Get best Val Acc
            std::vector<std::string> valepoch;
            for (int i = 0; i < text.size(); ++i) {
                std::string line = text[i];
                if (line.find("Val epoch") == std::string::npos) continue;
                if (!valepoch.empty()) {
                    int idx = valepoch.back().find("Val epoch");
                    std::string pre_epoch = std::string(valepoch.back().begin() + idx + 11, valepoch.back().begin() + idx + 13);
                    valepoch.push_back(line);
                    idx = valepoch.back().find("Val epoch");
                    std::string cur_epoch = std::string(valepoch.back().begin() + idx + 11, valepoch.back().begin() + idx + 13);
                    if (pre_epoch == cur_epoch) continue;
                    
                    std::string lastbatch = valepoch[valepoch.size() - 2];
                    for (int j = 0; j < lastbatch.size(); ++j) {
                        if (lastbatch[j] != 'A' || lastbatch[j + 1] != 'c' || lastbatch[j + 2] != 'c') continue;
                        BestRes[s][fold] = std::max(BestRes[s][fold], std::stod(std::string(lastbatch.begin() + j + 4, lastbatch.end())));
                    }
                    valepoch.clear();
                }
                valepoch.push_back(line);
            }
            // Get last Val Acc
            std::string lastLine = valepoch.back();
            for (int i = 0; i < lastLine.size(); ++i) {
                if (lastLine[i] != 'A' || lastLine[i + 1] != 'c' || lastLine[i + 2] != 'c') continue;
                LastRes[s][fold] = std::stod(std::string(lastLine.begin() + i + 4, lastLine.end()));
                BestRes[s][fold] = std::max(BestRes[s][fold], LastRes[s][fold]);
            }
        }
    }
    // Save to csv
    std::string saveFile = sum_save + "\\LastResSummary.csv";
    std::ofstream outfile(saveFile);
    if (!outfile.is_open()) {
        std::cerr << "Unable to open file" << std::endl;
        return;
    }
    for (int s = 1; s <= 20; ++s) {
        for (int fold = 1; fold <= 5; ++fold) {
            if (fold > 1) outfile << ",";
            outfile << LastRes[s][fold];
        }
        outfile << std::endl;
    }
    outfile.close();

    saveFile = sum_save + "\\BestResSummary.csv";
    outfile = std::ofstream(saveFile);
    if (!outfile.is_open()) {
        std::cerr << "Unable to open file" << std::endl;
        return;
    }
    for (int s = 1; s <= 20; ++s) {
        for (int fold = 1; fold <= 5; ++fold) {
            if (fold > 1) outfile << ",";
            outfile << BestRes[s][fold];
        }
        outfile << std::endl;
    }
    outfile.close();

    saveFile = sum_save + "\\BestResMacroPSummary.csv";
    outfile = std::ofstream(saveFile);
    if (!outfile.is_open()) {
        std::cerr << "Unable to open file" << std::endl;
        return;
    }
    for (int s = 1; s <= 20; ++s) {
        for (int fold = 1; fold <= 5; ++fold) {
            if (fold > 1) outfile << ",";
            outfile << BestResMacroP[s][fold];
        }
        outfile << std::endl;
    }
    outfile.close();

    saveFile = sum_save + "\\BestResMacroRSummary.csv";
    outfile = std::ofstream(saveFile);
    if (!outfile.is_open()) {
        std::cerr << "Unable to open file" << std::endl;
        return;
    }
    for (int s = 1; s <= 20; ++s) {
        for (int fold = 1; fold <= 5; ++fold) {
            if (fold > 1) outfile << ",";
            outfile << BestResMacroR[s][fold];
        }
        outfile << std::endl;
    }
    outfile.close();

    saveFile = sum_save + "\\BestResMacroF1Summary.csv";
    outfile = std::ofstream(saveFile);
    if (!outfile.is_open()) {
        std::cerr << "Unable to open file" << std::endl;
        return;
    }
    for (int s = 1; s <= 20; ++s) {
        for (int fold = 1; fold <= 5; ++fold) {
            if (fold > 1) outfile << ",";
            outfile << BestResMacroF1[s][fold];
        }
        outfile << std::endl;
    }
    outfile.close();
    return;
}

int64_t Utils::countParameters(const std::shared_ptr<torch::nn::Module> model) {
    int64_t totoalParameters = 0;
    for (const auto& param : model->parameters()) {
        totoalParameters += param.numel();
    }
    return totoalParameters;
}