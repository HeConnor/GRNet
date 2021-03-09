#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Software: PyCharm
# @Version : Python-
# @Author  : Shengji He
# @Email   : hsjbit@163.com
# @File    : test_hsj.py
# @Time    : 2021/3/8 11:37
# @Description:
import torch
import logging
from extensions.gridding import Gridding

import open3d
import numpy as np

import utils.data_loaders
import utils.helpers
from config import cfg
from tensorboardX import SummaryWriter
import os
from datetime import datetime


def main():
    output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', datetime.now().isoformat())
    cfg.DIR.LOGS = output_dir % 'logs'

    test_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    PARTIAL_POINTS_PATH = './datasets/kitti/cars/frame_0_car_0.pcd'
    BOUNDING_BOX_FILE_PATH = './datasets/kitti/bboxes/frame_0_car_0.txt'

    pc = open3d.io.read_point_cloud(PARTIAL_POINTS_PATH)
    # open3d.visualization.draw_geometries([pc])
    ptcloud = np.array(pc.points)
    ptcloud_img = utils.helpers.get_ptcloud_img(ptcloud)
    test_writer.add_image('Model%02d/InitialImg' % 1, ptcloud_img, 0)

    print('{}'.format(ptcloud.shape))

    # cfg.DATASET.TEST_DATASET = 'KITTI'
    # cfg.DATASETS.KITTI.CATEGORY_FILE_PATH = './datasets/KITTI.json'
    # cfg.DATASETS.KITTI.PARTIAL_POINTS_PATH = './datasets/kitti/cars/%s.pcd'
    # cfg.DATASETS.KITTI.BOUNDING_BOX_FILE_PATH = './datasets/kitti/bboxes/%s.txt'
    # cfg.CONST.NUM_WORKERS = 1

    torch.backends.cudnn.benchmark = True

    dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST),
        batch_size=1,
        num_workers=cfg.CONST.NUM_WORKERS,
        collate_fn=utils.data_loaders.collate_fn,
        pin_memory=True,
        shuffle=False)

    gridding = Gridding(scale=64)
    train_on_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if train_on_gpu else "cpu")
    gridding = gridding.to(device)

    taxonomy_id, model_id, data = test_data_loader.dataset[0]

    taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
    model_id = model_id[0]
    for k, v in data.items():
        # data[k] = utils.helpers.var_or_cuda(v)
        data[k] = v.to(device)

    partial_cloud = data['partial_cloud']
    partial_cloud = partial_cloud.unsqueeze(0)
    pt_features_64_l = gridding(partial_cloud).view(-1, 1, 64, 64, 64)

    pt_features_cpu_64_l = pt_features_64_l.squeeze().cpu().numpy()
    np.save('gridding.npy', pt_features_cpu_64_l)

    sparse_ptcloud = partial_cloud.squeeze().cpu().numpy()
    sparse_ptcloud_img = utils.helpers.get_ptcloud_img(sparse_ptcloud)
    test_writer.add_image('Model%02d/SparseReconstruction' % 1, sparse_ptcloud_img, 0)
    # for model_idx, (taxonomy_id, model_id, data) in enumerate(test_data_loader):
    # taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
    # model_id = model_id[0]
    # for k, v in data.items():
    #     data[k] = utils.helpers.var_or_cuda(v)
    #
    #
    #
    # partial_cloud = data['partial_cloud']
    # pt_features_64_l = gridding(partial_cloud).view(-1, 1, 64, 64, 64)
    #
    # pt_features_cpu_64_l = pt_features_64_l.squeeze().cpu().numpy()
    # np.save('gridding.npy', pt_features_cpu_64_l)
    #
    # sparse_ptcloud = partial_cloud.squeeze().cpu().numpy()
    # sparse_ptcloud_img = utils.helpers.get_ptcloud_img(sparse_ptcloud)
    # test_writer.add_image('Model%02d/SparseReconstruction' % 1, sparse_ptcloud_img, 0)

    # partial_cloud = data['partial_cloud']
    # pt_features_64_l = self.gridding(partial_cloud).view(-1, 1, 64, 64, 64)
    test_writer.close()
    pass


def main_gridding():
    import matplotlib.pyplot as plt
    data = np.load('gridding.npy')
    for i in range(64):
        plt.imshow(data[i, :, :])
        plt.show()
        plt.clf()


if __name__ == '__main__':
    logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.DEBUG)
    main()
    # main_gridding()
    print('done')
