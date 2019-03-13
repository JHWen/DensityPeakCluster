#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import math
import logging
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import plot_scatter_diagram

logger = logging.getLogger("dpc_cluster")


def plot_rho_delta(rho, delta):
    """
    Plot scatter diagram for rho-delta points

    Args:
        rho   : rho list
        delta : delta list
    """
    logger.info("PLOT: rho-delta plot")
    plot_scatter_diagram(0, rho[1:], delta[1:], x_label='rho', y_label='delta', title='Decision Graph')
    plt.savefig('Decision Graph.jpg')
    plt.show()


def load_paper_data(distance_f):
    """
    Load distance from data

    Args:
        distance_f : distance file, the format is column1-index 1, column2-index 2, column3-distance

    Returns:
        distances dict, max distance, min distance, max continues id
    """
    logger.info("PROGRESS: load data")
    distances = {}
    min_dis, max_dis = sys.float_info.max, 0.0
    max_id = 0
    with open(distance_f, 'r') as fp:
        for line in fp:
            x1, x2, d = line.strip().split(' ')
            x1, x2 = int(x1), int(x2)
            max_id = max(max_id, x1, x2)
            dis = float(d)
            min_dis, max_dis = min(min_dis, dis), max(max_dis, dis)
            distances[(x1, x2)] = float(d)
            distances[(x2, x1)] = float(d)
    for i in range(max_id + 1):
        distances[(i, i)] = 0.0
    logger.info("max_id:" + str(max_id))
    logger.info("PROGRESS: load end")
    return distances, max_dis, min_dis, max_id


def select_dc(max_id, max_dis, min_dis, distances, auto=False):
    """
    Select the local density threshold, default is the method used in paper, auto is `auto_select_dc`

    Args:
        max_id    : max continues id
        max_dis   : max distance for all points
        min_dis   : min distance for all points
        distances : distance dict
        auto      : use auto dc select or not

    Returns:
        dc that local density threshold
    """
    logger.info("PROGRESS: select dc")
    if auto:
        return autoselect_dc(max_id, max_dis, min_dis, distances)
    percent = 2.0
    position = int(max_id * (max_id + 1) / 2 * percent / 100)
    dc = sorted(distances.values())[position * 2 + max_id]
    logger.info("PROGRESS: dc - " + str(dc))
    return dc


def autoselect_dc(max_id, max_dis, min_dis, distances):
    """
    Auto select the local density threshold that let average neighbor is 1-2 percent of all nodes.

    Args:
        max_id    : max continues id
        max_dis   : max distance for all points
        min_dis   : min distance for all points
        distances : distance dict

    Returns:
        dc that local density threshold
    """
    dc = (max_dis + min_dis) / 2

    while True:
        nneighs = sum([1 for v in distances.values() if v < dc]) / max_id ** 2
        if 0.01 <= nneighs <= 0.002:
            break
        # binary search
        if nneighs < 0.01:
            min_dis = dc
        else:
            max_dis = dc
        dc = (max_dis + min_dis) / 2
        if max_dis - min_dis < 0.0001:
            break
    return dc


def local_density(max_id, distances, dc, gauss=True, cutoff=False):
    """
    Compute all points' local density

    Args:
        max_id    : max continues id
        distances : distance dict  距离矩阵，以dict形式存储
        dc        : cutoff distance  截断距离
        gauss     : use gauss func or not(can't use together with cutoff)
        cutoff    : use cutoff func or not(can't use together with gauss)

    Returns:
        local density vector that index is the point index that start from 1 (numpy array)
    """
    assert gauss and not cutoff and gauss or cutoff
    logger.info("PROGRESS: compute local density")
    if gauss:
        func = gauss_func
    else:
        func = cutoff_func
    # id start from 1
    rho = [-1] + [0] * max_id

    for i in range(1, max_id):
        for j in range(i + 1, max_id + 1):
            rho[i] += func(distances[(i, j)], dc)
            rho[j] += func(distances[(i, j)], dc)
        if i % (max_id // 10) == 0:
            logger.info("PROGRESS: at index #%i" % i)
    return np.array(rho, np.float32)


# Gaussian kernel
def gauss_func(dij, dc):
    return math.exp(- (dij / dc) ** 2)


# cutoff kernel
def cutoff_func(dij, dc):
    return 1 if dij < dc else 0


def min_distance(max_id, max_dis, distances, rho):
    """
    Compute all points' min distance to the higher local density point(which is the nearest neighbor)

    Args:
        max_id    : max continues id
        max_dis   : max distance for all points
        distances : distance dict
        rho       : local density vector that index is the point index that start from 1

    Returns:
        min_distance vector, nearest neighbor vector
    """
    logger.info("PROGRESS: compute min distance to nearest higher density neigh")
    # 降序排序
    sort_rho_idx = np.argsort(-rho)
    delta, nearest_neighbor = [0.0] + [float(max_dis)] * (len(rho) - 1), [0] * len(rho)
    delta[sort_rho_idx[0]] = -1.
    # sort_rho_idx列表中最后一个元素是-1对应的下标
    for i in range(1, max_id):
        for j in range(0, i):
            old_i, old_j = sort_rho_idx[i], sort_rho_idx[j]
            if distances[(old_i, old_j)] <= delta[old_i]:
                delta[old_i] = distances[(old_i, old_j)]
                nearest_neighbor[old_i] = old_j
        if i % (max_id // 10) == 0:
            logger.info("PROGRESS: at index #%i" % i)
    # todo 为啥取delta的最大值,貌似只要是个最大值就好了
    delta[sort_rho_idx[0]] = max(delta)
    return np.array(delta, np.float32), np.array(nearest_neighbor, np.int)


class DensityPeakCluster(object):
    def local_density(self, load_func, distance_f, dc=None, auto_select_dc=False):
        """
        Just compute local density

        Args:
            load_func     : load func to load data
            distance_f    : distance data file
            dc            : local density threshold, call select_dc if dc is None
            auto_select_dc : auto select dc or not

        Returns:
            distances dict, max distance, min distance, max index(最大的id), local density vector, dc
        """
        assert not (dc is not None and auto_select_dc)
        distances, max_dis, min_dis, max_id = load_func(distance_f)
        if dc is None:
            dc = select_dc(max_id, max_dis, min_dis, distances, auto=auto_select_dc)
        rho = local_density(max_id, distances, dc)
        return distances, max_dis, min_dis, max_id, rho, dc

    # todo 查看聚类方法
    def cluster(self, load_func, distance_f, density_threshold, distance_threshold, dc=None, auto_select_dc=False):
        """
        Cluster the data

        Args:
            load_func          : load func to load data
            distance_f         : distance data file
            dc                 : local density threshold, call select_dc if dc is None
            density_threshold  : local density threshold for choosing cluster center
            distance_threshold : min distance threshold for choosing cluster center
            auto_select_dc      : auto select dc or not

        Returns:
            local density vector, min_distance vector, nearest neighbor vector
        """
        assert not (dc is not None and auto_select_dc)
        # 1.calculate rho(local density)
        distances, max_dis, min_dis, max_id, rho, dc = self.local_density(load_func, distance_f, dc=dc,
                                                                          auto_select_dc=auto_select_dc)

        # 2.calculate delta (minimum distance between point i and any other point with higher density)
        delta, nearest_neighbor = min_distance(max_id, max_dis, distances, rho)
        logger.info("PROGRESS: start cluster")

        # 绘制决策图
        plot_rho_delta(rho, delta)
        # 3.通过设定rho和delta阈值来确定聚类中心个数
        # cl/icl in cluster_dp.m
        cluster, cluster_center = {}, {}
        for idx, (ldensity, mdistance, neighbor_item) in enumerate(zip(rho, delta, nearest_neighbor)):
            if idx == 0:
                continue
            if ldensity >= density_threshold and mdistance >= distance_threshold:
                cluster_center[idx] = idx
                cluster[idx] = idx
            else:
                cluster[idx] = -1

        # 4.非聚类中心数据点归类，归类给最近的邻居的聚类中心就好啦，这里用了nearest_neighbor
        # assignation
        ordrho = np.argsort(-rho)
        for i in range(ordrho.shape[0] - 1):
            if ordrho[i] == 0:
                continue
            if cluster[ordrho[i]] == -1:
                cluster[ordrho[i]] = cluster[nearest_neighbor[ordrho[i]]]
            if i % (max_id // 10) == 0:
                logger.info("PROGRESS: at index #%i" % i)

        # 5.聚类中心多于1个的话，对聚类中心划分cluster core和cluster halo
        # halo : cluster core 和 cluster halo的标志
        halo, bord_rho = {}, {}

        for i in range(1, ordrho.shape[0]):
            halo[i] = cluster[i]

        if len(cluster_center) > 0:
            for idx in cluster_center.keys():
                bord_rho[idx] = 0.0
            # 分别计算这几个聚类平均密度
            # 1~n-1
            for i in range(1, rho.shape[0] - 1):
                # i+1~n
                for j in range(i + 1, rho.shape[0]):
                    if cluster[i] != cluster[j] and distances[i, j] <= dc:
                        rho_aver = (rho[i] + rho[j]) / 2.0
                        if rho_aver > bord_rho[cluster[i]]:
                            bord_rho[cluster[i]] = rho_aver
                        if rho_aver > bord_rho[cluster[j]]:
                            bord_rho[cluster[j]] = rho_aver

            for i in range(1, rho.shape[0]):
                if rho[i] < bord_rho[cluster[i]]:
                    halo[i] = 0
        for i in range(1, rho.shape[0]):
            if halo[i] == 0:
                cluster[i] = - 1

        self.cluster, self.cluster_center = cluster, cluster_center
        self.distances = distances
        self.max_id = max_id
        logger.info("PROGRESS: ended")
        return rho, delta, nearest_neighbor
