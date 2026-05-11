#include "Metrics.hpp"

Metrics calculateMetrics(const torch::Tensor& y_true, const torch::Tensor& y_pred, int num_classes) {
    torch::Tensor cm = torch::zeros({num_classes, num_classes}, torch::kInt64);
    for (int i = 0; i < y_true.size(0); ++i) {
        int t = y_true[i].item<int64_t>();
        int p = y_pred[i].item<int64_t>();
        cm[t][p] = cm[t][p] + 1;
    }

    std::vector<float> tp(num_classes, 0.0f);
    std::vector<float> fp(num_classes, 0.0f);
    std::vector<float> fn(num_classes, 0.0f);

    for (int c = 0; c < num_classes; ++c) {
        tp[c] = cm[c][c].item<float>();
        fn[c] = cm[c].sum().item<float>() - tp[c];
        fp[c] = cm.slice(1, c, c + 1).sum().item<float>() - tp[c];
    }

    float correct = cm.trace().item<float>();
    float total = y_true.size(0);
    float acc = correct / total;

    std::vector<float> p_list, r_list, f1_list;
    for(int c = 0; c < num_classes; ++c) {
        float precision = 0.0f;
        if (tp[c] + fp[c] > 0) precision = tp[c] / (tp[c] + fp[c]);

        float recall = 0.0f;
        if (tp[c] + fn[c] > 0) recall = tp[c] / (tp[c] + fn[c]);

        float f1 = 0.0f;
        if (precision + recall > 0) f1 = 2 * precision * recall / (precision + recall);

        p_list.push_back(precision);
        r_list.push_back(recall);
        f1_list.push_back(f1);
    }

    auto mean = [](const std::vector<float>& v) {
        float sum = 0;
        for (auto x : v) sum += x;
        return sum / v.size();
    };

    float macro_p = mean(p_list);
    float macro_r = mean(r_list);
    float macro_f1 = mean(f1_list);

    return {acc, macro_p, macro_r, macro_f1};
}