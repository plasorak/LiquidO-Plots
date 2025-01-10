from brokenaxes import brokenaxes
import click
from common import CommonParameters, alphabet, drop_hits
import copy
import json
import logging
import matplotlib
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
from rich.logging import RichHandler
import sys
from time_cluster_maker import Cluster, TimeClusterMaker
import uproot

epsilon = 1e-6
logger = logging.getLogger("make_timing_plot")

class TimingPlotParameters:
    def __init__(self, plot_options):
        self.common = CommonParameters(plot_options)
        self.legend_position = plot_options.get("legend_position", "best")
        self.time_merge_ns = plot_options.get("time_merge_ns", 50)


def merge_timing_intervals(time_spans, time_merge_ns):

    new_time_spans = []
    local_new_time_spans = time_spans

    while True:
        time_spans_1 = local_new_time_spans

        for_deletion = []

        for i, ts1 in enumerate(time_spans_1):
            if ts1[0] > ts1[1]:
                raise ValueError(f'Time span {ts1} is invalid')

            for j, ts2 in enumerate(time_spans_1):

                new_interval = []
                if ts2[0] > ts2[1]:
                    raise ValueError(f'Time span {ts2} is invalid')

                if abs(ts1[0] - ts2[0]) < epsilon and abs(ts1[1] - ts2[1]) < epsilon: # skip if the same interval
                    continue

                elif ts1[0] < ts2[0] and ts2[1] < ts1[1]: # if ts2 is in ts1
                    new_interval = ts1
                    del local_new_time_spans[j]
                    for_deletion.append(ts1)
                    logger.debug(f'ts2 is in ts1: {ts1} {ts2}')


                elif ts1[0] < ts2[0] and ts2[0] < ts1[1]: # if the intervals overlap
                    new_interval = [min(ts1[0], ts2[0]), max(ts1[1], ts2[1])]
                    del local_new_time_spans[j]
                    for_deletion.append(ts1)
                    logger.debug(f'Overlap: {ts1} {ts2}')


                elif abs(ts1[1] - ts2[0]) < time_merge_ns: # if the intervals are close to each other
                    del local_new_time_spans[j]
                    for_deletion.append(ts1)
                    new_interval = [min(ts1[0], ts2[0]), max(ts1[1], ts2[1])]
                    logger.debug(f'Close by: {ts1} {ts2}')


                else:
                    logger.debug(f'No overlap: {ts1} {ts2}')
                    new_interval = ts1

                if new_interval != [] and new_interval not in local_new_time_spans:
                    logger.debug(f'Adding new interval: {new_interval}')
                    local_new_time_spans.append(new_interval)

        for ts in for_deletion:
            local_new_time_spans.remove(ts)

        if new_time_spans == local_new_time_spans:
            break
        else:
            new_time_spans = local_new_time_spans

    new_time_spans = sorted(new_time_spans, key=lambda x: x[0])
    return new_time_spans

@click.command()
@click.argument('input_hit_data', type=click.Path(exists=True))
@click.argument('input_cluster_data', type=click.Path(exists=True))
@click.argument('output', type=click.Path(exists=False))
@click.option('--plot-options', type=click.Path(exists=True), default=None)
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']), default="INFO", help="Logging level")
def main(input_hit_data, input_cluster_data, output, plot_options, log_level):

    logging.basicConfig(format='%(message)s', handlers=[RichHandler(show_time=False)])
    logger.setLevel(log_level.upper())

    if plot_options is not None:

        option_key = input_hit_data.replace('/eos/user/p/plasorak/LiquidO/', '')
        rprint(f'Using option key: {option_key}')
        with open(plot_options, 'r') as plot_options_file:
            plot_options = json.load(plot_options_file)
        plot_options = plot_options.get(option_key, {})
        if plot_options != {}:
            rprint(f'Using plot options: {json.dumps(plot_options, indent=4)}')
        else:
            rprint(f'[red bold]No plot options found for {option_key}[/]')
    else:
        plot_options = {}

    cluster_data = {}
    with open(input_cluster_data, 'r') as f:
        cluster_data = json.load(f)

    parameters = TimingPlotParameters(plot_options)
    hit_data = uproot.open(input_hit_data)['op_hits'] .arrays(library="pd")

    if   parameters.common.view == 'xz':
        hit_data = hit_data[hit_data['h_is_xz']]
    elif parameters.common.view == 'yz':
        hit_data = hit_data[hit_data['h_is_yz']]

    hit_data = drop_hits(
        hit_data,
        parameters.common.hit_survival_prob,
        parameters.common.seed
    )

    hit_t = np.array(hit_data['h_time'])
    hit_x_key = "h_pos_z"
    hit_y_key = "h_pos_x" if parameters.common.view == 'xz' else 'h_pos_y'

    hit_x = np.array(hit_data[hit_x_key])
    hit_y = np.array(hit_data[hit_y_key])
    hit_t = np.array(hit_data['h_time'])

    clusters = {
        label: Cluster.get_from_data(
            value['x_min'], value['x_max'],
            value['y_min'], value['y_max'],
            value['t_min'], value['t_max'],
            hit_x, hit_y, hit_t
        ) for label, value in cluster_data.items()
    }

    cluster_titles = {k: v['title'] for k, v in cluster_data.items()}

    timing_intervals = [[c.t_min, c.t_max] for c in clusters.values()]

    for i in range(len(timing_intervals)):
        logger.info(f'{i}: {timing_intervals[i]}')

    logger.info('Merging...')
    timing_intervals = merge_timing_intervals(timing_intervals, parameters.time_merge_ns)

    for i in range(len(timing_intervals)):
        if i == 0:
            timing_intervals[i] = [0, int(timing_intervals[i][1]+parameters.time_merge_ns/2)+1]
        else:
            timing_intervals[i] = [int(timing_intervals[i][0]-parameters.time_merge_ns/2), int(timing_intervals[i][1]+parameters.time_merge_ns/2)+1]

    for i in range(len(timing_intervals)):
        logger.info(f'{i}: {timing_intervals[i]}')

    fig = plt.figure(figsize=(10,5))

    binning = np.array([])
    for i, ti in enumerate(timing_intervals):
        binning = np.concatenate([binning, np.arange(ti[0], ti[1], parameters.common.hit_lifetime_ns)])

    bax = brokenaxes(xlims=timing_intervals, hspace=.03)
    bax.hist(
        [clusters[k].hit_t for k in clusters.keys()],
        binning,
        log=True,
        histtype='bar',
        stacked=True,
        rasterized=True,
        label=[f'{k}: {cluster_titles.get(k, "")}' for k in clusters.keys()],
    )
    bax.set_xlabel('Time [ns]')
    bax.set_ylabel('Number of hits')
    bax.xaxis.set_ticks([])
    #secax = bax.secondary_yaxis()
    bax.legend()

    fig.savefig(output)

    # time_counts = np.zeros(len(time_binning)-1)
    # for cluster in clusters.values():
    #     counts, _ = np.histogram(cluster.hit_t, bins=time_binning)
    #     time_counts += counts

    # axtime[0].annotate('Time [ns]', (0.96,-0.15), xycoords='axes fraction')# transform=axtime[0].transAxes)
    # axtime[0].set_ylabel('Number of hits')
    # y_max = np.max(time_counts)*10
    # y_min = 0.5
    # axtime[0].set_ylim((y_min, y_max))
    # axtime[1].set_ylim((y_min, y_max))

    # s0 = b[0]
    # e0 = time_axis_cut[0]
    # s1 = time_axis_cut[1]
    # e1 = b[-1]

    # range_0 = e0-s0
    # range_1 = e1-s1
    # range_ = max(range_0, range_1)
    # s0 = s0 - 0.02 * range_
    # s1 = s1 - 0.02 * range_
    # e0 = s0 + range_
    # e1 = s1 + range_

    # axtime[0].set_xlim((s0, e0))
    # axtime[1].set_xlim((s1, e1))

    # axtime[0].spines['right'].set_visible(False)
    # axtime[1].spines['left'].set_visible(False)
    # axtime[0].yaxis.tick_left()
    # axtime[0].tick_params(labelright='off')
    # axtime[1].yaxis.set_tick_params(which='both',left=False)
    # # axtime[1].tick_params('y', which='both',  )
    # axtime[1].yaxis.set_ticks([])

    # d = .015  # how big to make the diagonal lines in axes coordinates
    # # arguments to pass plot, just so we don't keep repeating them
    # kwargs = dict(transform=axtime[0].transAxes, color='k', clip_on=False, linewidth=1)
    # axtime[0].plot((1-d, 1+d), (-d*2, +d*2), **kwargs)
    # # axtime[0].plot((1-d, 1+d), (1-d*2, 1+d*2), **kwargs)

    # kwargs.update(transform=axtime[1].transAxes)  # switch to the bottom axes
    # # axtime[1].plot((-d, +d), (1-d*2, 1+d*2), **kwargs)
    # axtime[1].plot((-d, +d), (-d*2, +d*2), **kwargs)


    # handles, labels = axtime[0].get_legend_handles_labels()
    # sorted_labels = sorted(labels)
    # sorted_handles = [h for _, h in sorted(zip(labels, handles))]
    # axtime[1].legend(sorted_handles, sorted_labels)



if __name__ == "__main__":
    main()

