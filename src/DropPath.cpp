#include "DropPath.hpp"

DropPathImpl::DropPathImpl(double drop_prob) {
    this->drop_prob = drop_prob;
    return;
}

torch::Tensor DropPathImpl::forward(torch::Tensor x) {
    if (drop_prob == 0. || !this->is_training()) {
        return x.clone();
    }

    auto shape = x.sizes();
    auto noise = torch::rand({shape[0]}, torch::device(x.device()).dtype(x.dtype()));
    auto keep_prob = 1 - drop_prob;
    auto mask = (noise < keep_prob).view({shape[0], 1, 1, 1});
    x = x.div(keep_prob);
    x = x * mask;
    return x;
}