#pragma once

#include <vector>
#include "../../common/torch_helper.h"

std::vector<at::Tensor> grid_subsampling_with_labels(
    at::Tensor points,
    at::Tensor labels,
    at::Tensor lengths,
    float voxel_size
);
