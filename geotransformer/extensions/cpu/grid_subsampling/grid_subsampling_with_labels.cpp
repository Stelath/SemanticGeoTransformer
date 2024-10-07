#include <cstring>
#include "grid_subsampling_with_labels.h"
#include "grid_subsampling_with_labels_cpu.h"

std::vector<at::Tensor> grid_subsampling_with_labels(
    at::Tensor points,
    at::Tensor labels,
    at::Tensor lengths,
    float voxel_size
) {
    CHECK_CPU(points);
    CHECK_CPU(labels);
    CHECK_CPU(lengths);
    CHECK_IS_FLOAT(points);
    CHECK_IS_LONG(labels);
    CHECK_IS_LONG(lengths);
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(labels);
    CHECK_CONTIGUOUS(lengths);

    std::size_t batch_size = lengths.size(0);
    std::size_t total_points = points.size(0);

    std::vector<PointXYZ> vec_points = std::vector<PointXYZ>(
        reinterpret_cast<PointXYZ*>(points.data_ptr<float>()),
        reinterpret_cast<PointXYZ*>(points.data_ptr<float>()) + total_points
    );
    std::vector<PointXYZ> vec_s_points;

    std::vector<long> vec_labels = std::vector<long>(
        labels.data_ptr<long>(),
        labels.data_ptr<long>() + total_points
    );
    std::vector<long> vec_s_labels;

    std::vector<long> vec_lengths = std::vector<long>(
        lengths.data_ptr<long>(),
        lengths.data_ptr<long>() + batch_size
    );
    std::vector<long> vec_s_lengths;

    grid_subsampling_with_labels_cpu(
        vec_points,
        vec_s_points,
        vec_labels,
        vec_s_labels,
        vec_lengths,
        vec_s_lengths,
        voxel_size
    );

    std::size_t total_s_points = vec_s_points.size();
    at::Tensor s_points = torch::zeros(
        {static_cast<long>(total_s_points), 3},
        at::device(points.device()).dtype(at::ScalarType::Float)
    );
    at::Tensor s_labels = torch::zeros(
        {static_cast<long>(total_s_points)},
        at::device(labels.device()).dtype(at::ScalarType::Long)
    );
    at::Tensor s_lengths = torch::zeros(
        {static_cast<long>(batch_size)},
        at::device(lengths.device()).dtype(at::ScalarType::Long)
    );

    std::memcpy(
        s_points.data_ptr<float>(),
        reinterpret_cast<float*>(vec_s_points.data()),
        sizeof(float) * total_s_points * 3
    );
    std::memcpy(
        s_labels.data_ptr<long>(),
        vec_s_labels.data(),
        sizeof(long) * total_s_points
    );
    std::memcpy(
        s_lengths.data_ptr<long>(),
        vec_s_lengths.data(),
        sizeof(long) * batch_size
    );

    return {s_points, s_labels, s_lengths};
}
