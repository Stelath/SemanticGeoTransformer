import os
import numpy as np

import torch

from config import make_cfg

from geotransformer.datasets.registration.threedmatch.dataset import ThreeDMatchPairDataset
from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)

def test_data_loader(cfg, benchmark):
    train_dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
        use_augmentation=cfg.train.use_augmentation,
        augmentation_noise=cfg.train.augmentation_noise,
        augmentation_rotation=cfg.train.augmentation_rotation,
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )

    test_dataset = ThreeDMatchPairDataset(
        cfg.data.dataset_root,
        benchmark,
        point_limit=cfg.test.point_limit,
        use_augmentation=False,
    )
    # test_loader = build_dataloader_stack_mode(
    #     test_dataset,
    #     registration_collate_fn_stack_mode,
    #     cfg.backbone.num_stages,
    #     cfg.backbone.init_voxel_size,
    #     cfg.backbone.init_radius,
    #     neighbor_limits,
    #     batch_size=cfg.test.batch_size,
    #     num_workers=cfg.test.num_workers,
    #     shuffle=False,
    # )

    return test_dataset, neighbor_limits

def main():
    cfg = make_cfg()
    test_dataset, neighbor_limits = test_data_loader(cfg, '3DMatch')

    single_dataset_sample = test_dataset[1]
    print(single_dataset_sample.keys())
    print(single_dataset_sample['ref_frame'])
    print(single_dataset_sample['src_frame'])
    print(single_dataset_sample['ref_points'].shape)
    print(single_dataset_sample['src_points'].shape)
    print(single_dataset_sample['transform'])

if __name__ == '__main__':
    main()