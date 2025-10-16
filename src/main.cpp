#include <nlohmann/json.hpp>
#include "SummarizeResult.hpp"
#include "StarNet2503081200TV.hpp"
#include "StarNet2503081200TVA.hpp"
#include "StarNet2503132330TVA.hpp"
#include "StarNet2503141330TVA.hpp"
#include "StarNet2510152000TVA.hpp"

int main(int argc, char* argv[]) {
    // 检查命令行参数的数量
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <path_to_json_file>" << std::endl;
        return -1;
    }

    // 获取 JSON 文件路径
    std::string jsonFilePath = argv[1];

    // 打开 JSON 文件
    std::ifstream ifs(jsonFilePath);
    if (!ifs.is_open()) {
        std::cout << "Failed to open file: " << jsonFilePath << std::endl;
        return -1;
    }

    // 解析 JSON 文件到 json 对象中
    nlohmann::json js;
    try {
        ifs >> js;
    } catch (nlohmann::json::parse_error& e) {
        std::cout << "JSON parse error at byte " << e.byte << ": " << e.what() << std::endl;
        return -1;
    }
    
    // 关闭文件流
    ifs.close();

    // 访问 JSON 数据
    std::string model = js.value("model", "");

    // 选择模型为汇总结果
    if (model == "SummarizeResult") {
        SummarizeResult::summarize(js["resSave"].get<std::string>(), js["sumSave"].get<std::string>());
        return 0;
    }

    if (model == "StarNet2503081200") {
        StarNet2503081200TVBuilder builder;
        builder.setCUDA(js["cuda"].get<bool>())
            .setEpochs(js["epochs"].get<int>())
            .setWarmup(js["warmup"].get<int>())
            .setBatchSize(js["batchSize"].get<int>())
            .setLearningRate(js["learningRate"].get<double>())
            .setBeta1(js["beta1"].get<double>())
            .setBeta2(js["beta2"].get<double>())
            .setWeightDecay(js["weightDecay"].get<double>())
            .setRootOrig(js["rootOrig"].get<std::string>())
            .setRootEnve(js["rootEnve"].get<std::string>())
            .setRootMUAPP2P(js["rootMUAPP2P"].get<std::string>())
            .setRootComm(js["rootComm"].get<std::string>())
            .setSavePath(js["savePath"].get<std::string>())
            .setCombination(js["combination"].get<int>());
        builder.config.model_config.setEmbed(js["embed"].get<int>())
            .setDepths(js["depths"].get<std::vector<int>>());
        for (int fold = 1; fold <= 5; ++fold) {
            builder.setValFold(fold).build();
        }
        return 0;
    }

    if (model == "StarNet2503081200A") {
        StarNet2503081200TVABuilder builder;
        builder.setCUDA(js["cuda"].get<bool>())
            .setEpochs(js["epochs"].get<int>())
            .setWarmup(js["warmup"].get<int>())
            .setBatchSize(js["batchSize"].get<int>())
            .setLearningRate(js["learningRate"].get<double>())
            .setBeta1(js["beta1"].get<double>())
            .setBeta2(js["beta2"].get<double>())
            .setWeightDecay(js["weightDecay"].get<double>())
            .setRootOrig(js["rootOrig"].get<std::string>())
            .setRootEnve(js["rootEnve"].get<std::string>())
            .setRootMUAPP2P(js["rootMUAPP2P"].get<std::string>())
            .setRootComm(js["rootComm"].get<std::string>())
            .setSavePath(js["savePath"].get<std::string>())
            .setCombination(js["combination"].get<int>());
        builder.config.model_config.setEmbed(js["embed"].get<int>())
            .setDepths(js["depths"].get<std::vector<int>>());
        for (int fold = 1; fold <= 5; ++fold) {
            builder.setValFold(fold).build();
        }
        return 0;
    }

    if (model == "StarNet2503132330A") {
        StarNet2503132330TVABuilder builder;
        builder.setCUDA(js["cuda"].get<bool>())
            .setEpochs(js["epochs"].get<int>())
            .setWarmup(js["warmup"].get<int>())
            .setBatchSize(js["batchSize"].get<int>())
            .setLearningRate(js["learningRate"].get<double>())
            .setBeta1(js["beta1"].get<double>())
            .setBeta2(js["beta2"].get<double>())
            .setWeightDecay(js["weightDecay"].get<double>())
            .setRootOrig(js["rootOrig"].get<std::string>())
            .setRootEnve(js["rootEnve"].get<std::string>())
            .setRootMUAPP2P(js["rootMUAPP2P"].get<std::string>())
            .setRootComm(js["rootComm"].get<std::string>())
            .setSavePath(js["savePath"].get<std::string>())
            .setCombination(js["combination"].get<int>());
        builder.config.model_config.setInChannels(js["inChannels"].get<int>())
            .setUpsampleSize(js["upsampleSize"].get<int>())
            .setDechannels(js["dechannels"].get<int>())
            .setDownsampleRatio(js["downsampleRatio"].get<int>())
            .setEmbed(js["embed"].get<int>())
            .setDepths(js["depths"].get<std::vector<int>>());
        for (int fold = 1; fold <= 5; ++fold) {
            builder.setValFold(fold).build();
        }
        return 0;
    }

    if (model == "StarNet2503141330A") {
        StarNet2503141330TVABuilder builder;
        builder.setCUDA(js["cuda"].get<bool>())
            .setEpochs(js["epochs"].get<int>())
            .setWarmup(js["warmup"].get<int>())
            .setBatchSize(js["batchSize"].get<int>())
            .setLearningRate(js["learningRate"].get<double>())
            .setBeta1(js["beta1"].get<double>())
            .setBeta2(js["beta2"].get<double>())
            .setWeightDecay(js["weightDecay"].get<double>())
            .setRootOrig(js["rootOrig"].get<std::string>())
            .setRootEnve(js["rootEnve"].get<std::string>())
            .setRootMUAPP2P(js["rootMUAPP2P"].get<std::string>())
            .setRootComm(js["rootComm"].get<std::string>())
            .setSavePath(js["savePath"].get<std::string>())
            .setCombination(js["combination"].get<int>());
        builder.config.model_config.setInChannels(js["inChannels"].get<int>())
            .setUpsampleSize(js["upsampleSize"].get<int>())
            .setDechannels(js["dechannels"].get<int>())
            .setEmbed(js["embed"].get<int>())
            .setDepths(js["depths"].get<std::vector<int>>());
        for (int fold = 1; fold <= 5; ++fold) {
            builder.setValFold(fold).build();
        }
        return 0;
    }

    if (model == "StarNet2510152000A") {
        StarNet2510152000TVABuilder builder;
        builder.setCUDA(js["cuda"].get<bool>())
            .setEpochs(js["epochs"].get<int>())
            .setWarmup(js["warmup"].get<int>())
            .setBatchSize(js["batchSize"].get<int>())
            .setLearningRate(js["learningRate"].get<double>())
            .setBeta1(js["beta1"].get<double>())
            .setBeta2(js["beta2"].get<double>())
            .setWeightDecay(js["weightDecay"].get<double>())
            .setRootOrig(js["rootOrig"].get<std::string>())
            .setRootEnve(js["rootEnve"].get<std::string>())
            .setRootMUAPP2P(js["rootMUAPP2P"].get<std::string>())
            .setRootComm(js["rootComm"].get<std::string>())
            .setSavePath(js["savePath"].get<std::string>())
            .setCombination(js["combination"].get<int>());
        builder.config.model_config.setEmbed(js["embed"].get<int>())
            .setDepths(js["depths"].get<std::vector<int>>());
        for (int fold = 1; fold <= 5; ++fold) {
            builder.setValFold(fold).build();
        }
        return 0;
    }

    return 0;
}