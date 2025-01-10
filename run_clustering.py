import click
from common import CommonParameters, alphabet, drop_hits
from dbscan import DBScan
import json
import matplotlib
import matplotlib.animation as animation
from matplotlib import colors
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, TextArea, VPacker
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
import pandas
import pandas as pd
from particle import Particle
from rich import print as rprint
from rich.table import Table
import sys
from time_cluster_maker import Cluster, TimeClusterMaker
import uproot

class ClusteringParameters:
    def __init__(self, plot_options):
        self.common = CommonParameters(plot_options)
        self.db_scan_params = plot_options.get('db_scan_params', {
            "eps": 400,
            "min_samples": 5
        })
        self.merge_cluster = plot_options.get('merge_cluster', [])
        self.min_hits = plot_options.get('min_hits', 10)


@click.command()
@click.argument('input_data', type=click.Path(exists=True))
@click.argument('output', type=click.Path(exists=False))
@click.option('--options', type=click.Path(exists=True), default=None)
def main(input_data, output, options):

    if not output.endswith('.json'):
        rprint(f'[red bold]Output file must have a .json extension[/]')
        return

    if options is not None:

        option_key = input_data.replace('/eos/user/p/plasorak/LiquidO/', '')
        rprint(f'Using option key: {option_key}')
        with open(options, 'r') as plot_options_file:
            options = json.load(options_file)
        options = options.get(option_key, {})
        if options != {}:
            rprint(f'Using plot options: {json.dumps(options, indent=4)}')
        else:
            rprint(f'[red bold]No plot options found for {option_key}[/]')
    else:
        options = {}

    parameters = ClusteringParameters(options)

    hit_data   = uproot.open(input_data)['op_hits'] .arrays(library="pd")

    if   parameters.common.view == 'xz':
        hit_data = hit_data[hit_data['h_is_xz']]
    elif parameters.common.view == 'yz':
        hit_data = hit_data[hit_data['h_is_yz']]

    hit_data = drop_hits(
        hit_data,
        parameters.common.hit_survival_prob,
        parameters.common.seed
    )


    hit_x_key = "h_pos_z"
    hit_y_key = "h_pos_x" if parameters.common.view == 'xz' else 'h_pos_y'

    binning_x = np.arange(-7477.5, 7577.5, 30) if parameters.common.view == 'xz' else np.arange(-7477.5+15, 7577.5+15, 30)
    binning_y = np.arange(-2625,   2625,   15) if parameters.common.view == 'xz' else np.arange(-2640,      2640,      15)

    hit_x = np.array(hit_data[hit_x_key])
    hit_y = np.array(hit_data[hit_y_key])
    hit_t = np.array(hit_data['h_time'])

    time_binning = np.arange(0.,np.max(hit_data['h_time']), parameters.common.hit_lifetime_ns)

    hit_counts, xedges, yedges = np.histogram2d(hit_x, hit_y, bins=(binning_x, binning_y))

    x_indices = np.where(hit_counts>0)[0]
    y_indices = np.where(hit_counts>0)[1]
    clusterable_x = (xedges[x_indices]+xedges[x_indices+1])/2
    clusterable_y = (yedges[y_indices]+yedges[y_indices+1])/2

    dbscan = DBScan(
        eps=parameters.db_scan_params['eps'],
        min_hits=parameters.db_scan_params['min_samples'],
        hit_x=clusterable_x,
        hit_y=clusterable_y
    )
    dbscan.run()
    space_clusters = dbscan.clusters

    print(f'Number of space clusters before prunning: {len(space_clusters)}')
    space_clusters = [cluster for cluster in space_clusters if cluster.n_hits]
    print(f'Number of space clusters after prunning: {len(space_clusters)}')

    overall_index = 0

    clusters = {}

    for i_space, space_cluster in enumerate(space_clusters):

        x_min, x_max, y_min, y_max = space_cluster.get_min_max()

        hit_t_ = np.array(hit_t[np.where((hit_x>x_min)&(hit_x<x_max)&(hit_y>y_min)&(hit_y<y_max))[0]])

        time_counts, _ = np.histogram(hit_t_, bins=time_binning)

        time_cluster_maker = TimeClusterMaker(time_binning, time_counts, parameters.common.hit_lifetime_ns)
        time_cluster_maker.run_edge_detector()
        time_clusters = time_cluster_maker.clusters


        print(f'Number of time clusters before prunning: {len(time_clusters)}')
        time_clusters = [ cluster for cluster in time_clusters if cluster.n_hits>parameters.min_hits ]
        print(f'Number of time clusters after prunning: {len(time_clusters)}')

        for i_time, time_cluster in enumerate(time_clusters):
            t_min = time_cluster.start
            t_max = time_cluster.stop
            clusters[alphabet[overall_index]] = Cluster.get_from_data(x_min, x_max, y_min, y_max, t_min, t_max, hit_x, hit_y, hit_t)

            def space_reduce(cluster, eps, min_hits):
                hit_counts, xedges, yedges = np.histogram2d(cluster.hit_x, cluster.hit_y, bins=(binning_x, binning_y))

                x_indices = np.where(hit_counts>0)[0]
                y_indices = np.where(hit_counts>0)[1]
                clusterable_x = (xedges[x_indices]+xedges[x_indices+1])/2
                clusterable_y = (yedges[y_indices]+yedges[y_indices+1])/2

                dbscan = DBScan(
                    eps,
                    min_hits,
                    clusterable_x,
                    clusterable_y,
                )
                dbscan.run()
                clusters = dbscan.clusters
                max_cluster = max(clusters, key=lambda c: c.n_hits)
                print(f'{cluster.x_min=}, {cluster.x_max=}, {cluster.y_min=}, {cluster.y_max=}, {cluster.n_hits=}')
                cluster.x_min, cluster.x_max, cluster.y_min, cluster.y_max = max_cluster.get_min_max()
                cluster.n_hits = max_cluster.n_hits
                print(f'{cluster.x_min=}, {cluster.x_max=}, {cluster.y_min=}, {cluster.y_max=}, {cluster.n_hits=}')

            space_reduce(
                clusters[alphabet[overall_index]],
                eps=parameters.db_scan_params['eps'],
                min_hits=parameters.db_scan_params['min_samples']
            )

            overall_index += 1

    for merge in parameters.merge_cluster:
        clusters_to_merge = [cluster for label, cluster in clusters.items() if label in merge]
        new_label = merge[0]
        new_cluster = Cluster.union(clusters_to_merge)
        for label in merge:
            del clusters[label]
        clusters[new_label] = new_cluster

    n_decay_clusters = len(clusters) - 1
    rprint(f'after merging clustering: {n_decay_clusters=}')
    for cluster in clusters.values():
        rprint(cluster)

    clusters_data = {
        label: {
            'x_min': float(cluster.x_min),
            'x_max': float(cluster.x_max),
            'y_min': float(cluster.y_min),
            'y_max': float(cluster.y_max),
            't_min': float(cluster.t_min),
            't_max': float(cluster.t_max),
            'n_hits': int(cluster.n_hits),
            'title': '',
        }
        for label, cluster in clusters.items()
    }

    with open(output, 'w') as f:
        json.dump(clusters_data, f, indent=4)
    rprint(f"[green]Saved clusters data to {output}[/]")



if __name__ == "__main__":
    main()

