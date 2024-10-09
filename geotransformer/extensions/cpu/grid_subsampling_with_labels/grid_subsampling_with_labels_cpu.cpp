#include "grid_subsampling_with_labels_cpu.h"
#include <utility>
#include <functional>

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& pair) const {
        return std::hash<T1>()(pair.first) ^ (std::hash<T2>()(pair.second) << 1);
    }
};

void single_grid_subsampling_with_labels_cpu(
    std::vector<PointXYZ>& points,
    std::vector<PointXYZ>& s_points,
    std::vector<long>& labels,
    std::vector<long>& s_labels,
    float voxel_size
) {
    PointXYZ minCorner = min_point(points);
    PointXYZ maxCorner = max_point(points);
    PointXYZ originCorner = floor(minCorner * (1. / voxel_size)) * voxel_size;

    std::size_t sampleNX = static_cast<std::size_t>(
        floor((maxCorner.x - originCorner.x) / voxel_size) + 1
    );
    std::size_t sampleNY = static_cast<std::size_t>(
        floor((maxCorner.y - originCorner.y) / voxel_size) + 1
    );

    std::size_t iX = 0;
    std::size_t iY = 0;
    std::size_t iZ = 0;
    std::size_t mapIdx = 0;

    // Key: (voxel index, label)
    std::unordered_map<std::pair<std::size_t, long>, SampledData, pair_hash> data;

    for (std::size_t idx = 0; idx < points.size(); ++idx) {
        const auto& p = points[idx];
        long label = labels[idx];

        iX = static_cast<std::size_t>(floor((p.x - originCorner.x) / voxel_size));
        iY = static_cast<std::size_t>(floor((p.y - originCorner.y) / voxel_size));
        iZ = static_cast<std::size_t>(floor((p.z - originCorner.z) / voxel_size));
        mapIdx = iX + sampleNX * iY + sampleNX * sampleNY * iZ;

        std::pair<std::size_t, long> key = std::make_pair(mapIdx, label);

        if (data.find(key) == data.end()) {
            data.emplace(key, SampledData());
        }

        data[key].update(p);
    }

    s_points.reserve(data.size());
    s_labels.reserve(data.size());
    for (const auto& v : data) {
        s_points.push_back(v.second.point * (1.0 / v.second.count));
        s_labels.push_back(v.first.second); // label
    }
}

void grid_subsampling_with_labels_cpu(
    std::vector<PointXYZ>& points,
    std::vector<PointXYZ>& s_points,
    std::vector<long>& labels,
    std::vector<long>& s_labels,
    std::vector<long>& lengths,
    std::vector<long>& s_lengths,
    float voxel_size
) {
    std::size_t start_index = 0;
    std::size_t batch_size = lengths.size();
    for (std::size_t b = 0; b < batch_size; b++) {
        std::vector<PointXYZ> cur_points(
            points.begin() + start_index,
            points.begin() + start_index + lengths[b]
        );
        std::vector<long> cur_labels(
            labels.begin() + start_index,
            labels.begin() + start_index + lengths[b]
        );
        std::vector<PointXYZ> cur_s_points;
        std::vector<long> cur_s_labels;

        single_grid_subsampling_with_labels_cpu(
            cur_points, cur_s_points, cur_labels, cur_s_labels, voxel_size
        );

        s_points.insert(s_points.end(), cur_s_points.begin(), cur_s_points.end());
        s_labels.insert(s_labels.end(), cur_s_labels.begin(), cur_s_labels.end());
        s_lengths.push_back(static_cast<long>(cur_s_points.size()));

        start_index += lengths[b];
    }
}
