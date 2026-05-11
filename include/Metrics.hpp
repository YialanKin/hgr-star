#pragma once
#include <torch/torch.h>
#include <vector>
#include <cmath>
#include <algorithm>

struct Metrics {
    float acc;
    float macro_p;
    float macro_r;
    float macro_f1;
};


Metrics calculateMetrics(const torch::Tensor& y_true, const torch::Tensor& y_pred, int num_classes);