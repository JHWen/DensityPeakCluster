#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging
from plot import plot_cluster, plot_rho_delta, plot_rhodelta_rho
from cluster import DensityPeakCluster, load_paper_data


def plot(data, density_threshold, distance_threshold, auto_select_dc=False):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger('step2_cluster')
    dp_cluster = DensityPeakCluster()
    rho, delta, nearest_neigh = dp_cluster.cluster(load_paper_data, data, density_threshold, distance_threshold,
                                                   auto_select_dc=auto_select_dc)
    logger.info(str(len(dp_cluster.cluster_center)) + ' center as below')
    points_of_cluster = {}
    for idx, center in dp_cluster.cluster_center.items():
        logger.info('%d %f %f' % (idx, rho[center], delta[center]))
        points_of_cluster[center] = 0
    # 　统计每个聚类的核心数据点的数量
    for idx, center in dp_cluster.cluster.items():
        if center != -1:
            points_of_cluster[center] += 1
    logger.info('每个聚类中数据点的数量:' + str(points_of_cluster))
    plot_rho_delta(rho, delta)  # plot to choose the threthold
    plot_rhodelta_rho(rho, delta)
    plot_cluster(dp_cluster)


if __name__ == '__main__':
    # plot('./data/data_in_paper/example_distances.dat', 20, 0.1, False)
    # plot('./data/data_others/spiral_distance.dat',8,5,False)
    # plot('./data/data_others/aggregation_distance.dat',15,4.5,False)
    # plot('./data/data_others/flame_distance.dat', 4, 7, False)
    # plot('./data/data_others/jain_distance.dat', 12, 10, False)
    # ./data/spam_mail_data/spam_distance_1000.dat
    # plot('./data/spam_mail_data/spam_distance_1000.dat', 180, 0.45, False)
    # ./data/data_others/aggregation_cosine_distance.dat
    plot('./data/data_others/aggregation_cosine_distance.dat', 10, 0.01, False)
