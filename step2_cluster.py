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
    for idx, center in dp_cluster.cluster_center.items():
        logger.info('%d %f %f' % (idx, rho[center], delta[center]))
    plot_rho_delta(rho, delta)  # plot to choose the threthold
    plot_rhodelta_rho(rho, delta)
    plot_cluster(dp_cluster)


if __name__ == '__main__':
    # plot('./data/data_in_paper/example_distances.dat', 20, 0.1, False)
    # plot('./data/data_others/spiral_distance.dat',8,5,False)
    # plot('./data/data_others/aggregation_distance.dat',15,4.5,False)
    # plot('./data/data_others/flame_distance.dat', 4, 7, False)
    plot('./data/data_others/jain_distance.dat', 12, 10, False)
