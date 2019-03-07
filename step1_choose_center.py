#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from plot import *
from cluster import DensityPeakCluster, load_paper_data


def plot(data, auto_select_dc=False):
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    dp_cluster = DensityPeakCluster()
    # calculate local density
    distances, max_dis, min_dis, max_id, rho, rc = dp_cluster.local_density(load_paper_data, data,
                                                                            auto_select_dc=auto_select_dc)
    # calculate delta, minimum distance
    delta, nearest_neigh = min_distance(max_id, max_dis, distances, rho)
    # plot to choose the threshold
    plot_rho_delta(rho, delta)


if __name__ == '__main__':
    # plot('./data/data_in_paper/example_distances.dat')
    # plot('./data/data_others/spiral_distance.dat')
    # plot('./data/data_others/aggregation_distance.dat')
    # plot('./data/data_others/flame_distance.dat')
    plot('./data/data_others/jain_distance.dat')
