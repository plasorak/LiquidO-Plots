import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas
import uproot
from rich import print as rprint
import matplotlib.animation as animation
import sys
from particle import Particle
import click
import pandas as pd
from matplotlib.gridspec import GridSpec
import matplotlib

CB_color_cycle = ['#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3', '#377eb8',
                  '#999999', '#e41a1c', '#dede00']

#np.set_printoptions(threshold=sys.maxsize)

pd.options.mode.copy_on_write = True
particle_color = {}

class AnchoredHScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
    """ size: length of bar in data units
        extent : height of bar ends in axes units """
    def __init__(self, size=1, extent = 0.03, label="", loc=2, ax=None,
                 pad=0.4, borderpad=0.5, ppad = 0, sep=2, prop=None,
                 frameon=True, linekw={}, **kwargs):
        if not ax:
            ax = plt.gca()
        trans = ax.get_xaxis_transform()
        size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
        line = Line2D([0,size],[0,0], **linekw)
        vline1 = Line2D([0,0],[-extent/2.,extent/2.], **linekw)
        vline2 = Line2D([size,size],[-extent/2.,extent/2.], **linekw)
        size_bar.add_artist(line)
        size_bar.add_artist(vline1)
        size_bar.add_artist(vline2)
        txt = matplotlib.offsetbox.TextArea(label, minimumdescent=False)
        self.vpac = matplotlib.offsetbox.VPacker(children=[size_bar,txt],
                                 align="center", pad=ppad, sep=sep)
        matplotlib.offsetbox.AnchoredOffsetbox.__init__(self, loc, pad=pad,
                 borderpad=borderpad, child=self.vpac, prop=prop, frameon=frameon,
                 **kwargs)

def add_padding(left, right, side:str, padding:float=0.):
    if not side in ['left', 'right']:
        raise RuntimeError(f'"side" should be "left" or "right". You provided: "{side}"')
    # print(f'\n\n{left=}, {right=}, {side=}, {padding=}')

    if left>right:
        raise RuntimeError(f'"left" should be bigger than "right". You provided: {left=}, {right=}')

    width = right-left

    if   right>0 and side=="right": number = right + width * (padding / 100.)
    elif right<0 and side=="right": number = right + width * (padding / 100.)
    elif left >0 and side=="left" : number = left  - width * (padding / 100.)
    elif left <0 and side=='left' : number = left  - width * (padding / 100.)

    return number

def adjust_for_aspect_ratio(x_min, x_max, y_min, y_max, aspect_ratio):
    current_aspect_ratio = (x_max - x_min) / (y_max - y_min)
    if current_aspect_ratio > aspect_ratio:
        y_center = (y_max + y_min) / 2
        y_range = (x_max - x_min) / aspect_ratio
        y_min = y_center - y_range / 2
        y_max = y_center + y_range / 2
    else:
        x_center = (x_max + x_min) / 2
        x_range = (y_max - y_min) * aspect_ratio
        x_min = x_center - x_range / 2
        x_max = x_center + x_range / 2
    new_aspect_ratio = (x_max - x_min) / (y_max - y_min)
    #print(f'Aspect ratio: {current_aspect_ratio} -> {new_aspect_ratio}')
    return x_min, x_max, y_min, y_max


def get_norm(x_min, x_max, y_min, y_max):
    return np.sqrt((x_max - x_min)*(x_max - x_min) + (y_max - y_min)*(y_max - y_min))

def add_truth_particle(
    ax,
    hit_key_x, hit_key_y,
    truth_data,
    x_min, x_max,
    y_min, y_max,
    time_start, time_end,
    legend_label=None,
    plot_primaries=True,
    plot_particle_ids=[],
    ):

    global particle_color

    truth_key_x = hit_key_x.replace('h_pos_', 'i_pos_')
    truth_key_y = hit_key_y.replace('h_pos_', 'i_pos_')

    all_track_ids = np.unique(truth_data['track_id'])

    plotted_track_ids = []
    primary = []

    for tid in all_track_ids:
        track_data = truth_data[truth_data['track_id'] == tid]
        track_data.sort_values('i_time', inplace=True)
        xs = track_data.loc[:, truth_key_x]
        ys = track_data.loc[:, truth_key_y]

        primary = (track_data['parent_id'] == 0).any()

        keep = (primary and plot_primaries) or (tid in plot_particle_ids)

        if not keep:
            continue

        pdg = track_data.iloc[0]['i_particle']
        if primary:
            xs = pd.concat([pd.Series([0]), xs])
            ys = pd.concat([pd.Series([0]), ys])
        p = Particle.from_pdgid(pdg)
        e = np.max(track_data['i_E'])
        M = p.mass
        if M is None and abs(pdg) in [12, 14, 16]:
            M = 0

        ke = e - M

        primary_str = ' (primary)' if primary else ''
        legend_str = f' {legend_label}' if legend_label else ''
        rprint(f"Plotting {tid}: {pdg=} {M=} {ke=} {e=} {p.name=} {primary_str}")

        label = f'${p.latex_name}$ KE {ke:.1f} MeV{primary_str}{legend_str}' if p is not None else f'{pdg} E {e} MeV'

        if tid in particle_color:
            label = None

        lines = ax.plot(
            xs,
            ys,
            linewidth=2,
            label = label,
            color = particle_color[tid] if tid in particle_color else None,
        )
        particle_color[tid] = lines[0].get_c()


def plot(
    ax,
    label,
    title,
    hit_data,
    hit_key_x,
    hit_key_y,
    bin_x,
    bin_y,
    truth_data,
    label_x,
    label_y,
    time_start,
    time_end,
    x_min,
    x_max,
    y_min,
    y_max,
    plot_axes = False,
    with_underlay = True,
    aspect_ratio = 1,
    legend_label = None,
    plot_primaries = True,
    plot_particle_ids = [],
):
    rprint(f"Plotting {label} {title}, {plot_primaries=}, {plot_particle_ids=}")
    hit_data_present = hit_data.where(hit_data["h_time"]>time_start, inplace=False)# * hit_data["h_time"]<time_end
    hit_data_present = hit_data_present.where(hit_data_present["h_time"]<time_end, inplace=False)# * hit_data["h_time"]<time_end

    hit_counts, xedges, yedges = np.histogram2d(
        hit_data_present[hit_key_x],
        hit_data_present[hit_key_y],
        bins=(bin_x, bin_y)
    )
    hit_counts = hit_counts.T


    hit_data_past = hit_data.where(hit_data["h_time"]<time_start, inplace=False)

    if with_underlay:
        underlay_hit_counts, xedges, yedges = np.histogram2d(
            hit_data_past[hit_key_x],
            hit_data_past[hit_key_y],
            bins=(bin_x, bin_y)
        )
        underlay_hit_counts = underlay_hit_counts.T

    X, Y = np.meshgrid(xedges, yedges)

    underlay_colormesh = None
    if with_underlay:
        underlay_hit_counts = np.ma.masked_array(underlay_hit_counts, underlay_hit_counts<0.5)
        underlay_colormesh = ax.pcolormesh(X, Y, underlay_hit_counts, norm='log', rasterized=True, cmap='Greys', vmin=np.min(underlay_hit_counts), vmax=np.max(underlay_hit_counts)*10)

    hit_counts = np.ma.masked_array(hit_counts, hit_counts<0.5)
    i = ax.pcolormesh(X, Y, hit_counts, norm='log', rasterized=True, vmin=np.min(hit_counts), vmax=np.max(hit_counts))

    x_min = np.min(hit_data_present[hit_key_x]) if x_min is None else x_min
    x_max = np.max(hit_data_present[hit_key_x]) if x_max is None else x_max
    y_min = np.min(hit_data_present[hit_key_y]) if y_min is None else y_min
    y_max = np.max(hit_data_present[hit_key_y]) if y_max is None else y_max

    x_min_padded, x_max_padded, y_min_padded, y_max_padded = adjust_for_aspect_ratio(x_min, x_max, y_min, y_max, aspect_ratio)

    x_min = x_min_padded
    x_max = x_max_padded
    y_min = y_min_padded
    y_max = y_max_padded

    add_truth_particle(
        ax,
        hit_key_x, hit_key_y,
        truth_data,
        x_min, x_max,
        y_min, y_max,
        time_start, time_end,
        legend_label=legend_label,
        plot_primaries=plot_primaries,
        plot_particle_ids=plot_particle_ids,
    )

    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))

    if label is not None:

        ax.annotate(
            f'{label}: {title}' if title is not None else label,
            (0.01, 0.99) if label == "a" else (0.05, 0.95),
            xycoords='axes fraction',
            backgroundcolor=('white', 0.8),
            va='center'
        )

    if label_x is not None and plot_axes:
        ax.set_xlabel(f'{label_x} [mm]')

    if label_y is not None and plot_axes:
        ax.set_ylabel(f'{label_y} [mm]')

    if not plot_axes:
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

    ax.set_aspect('equal', adjustable='box')

    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    width = x_max - x_min
    size_args = (1000, '1 m')

    if width < 500:
        size_args = (100, '10 cm')
    elif width < 1000:
        size_args = (200, '20 cm')
    elif width < 2000:
        size_args = (500, '50 cm')

    scalebar = AnchoredSizeBar(
        ax.transData,
        *size_args,
        'lower left',
        pad=1,
        color='red',
        frameon=False,
        size_vertical=1)

    ax.add_artist(scalebar)

    return x_min, x_max, y_min, y_max

@click.command()
@click.argument('input_data', type=click.Path(exists=True))
@click.argument('output', type=click.Path(exists=False))
@click.option('--plot-options', type=click.Path(exists=True), default=None)
def main(input_data, output, plot_options):

    run_clustering = False

    if plot_options is not None:
        import json
        option_key = input_data.replace('/eos/user/p/plasorak/LiquidO/', '')
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

    hit_lifetime_ns   = plot_options.get('hit_lifetime_ns', 5)
    view              = plot_options.get('view', 'xz')
    plot_axes         = plot_options.get('plot_axes', False)
    run_clustering    = plot_options.get('run_clustering', True)
    merge_cluster     = plot_options.get('merge_cluster', [])
    cluster_reindex   = plot_options.get('cluster_reindex', True)
    hit_survival_prob = plot_options.get('hit_survival_prob', 0.5)
    seed              = plot_options.get('seed', 123456)
    db_scan_params    = plot_options.get('db_scan_params', {
            "eps": 400,
            "min_samples": 5
        }
    )
    cluster_titles    = plot_options.get('cluster_titles', {})
    plot_particle_ids = plot_options.get('plot_particle_ids', {})
    plot_primaries    = plot_options.get('plot_primaries', {})

    hit_data   = uproot.open(input_data)['op_hits'] .arrays(library="pd")
    truth_data = uproot.open(input_data)['mc_truth'].arrays(library="pd")

    if   view == 'xz':
        hit_data = hit_data[hit_data['h_is_xz']]
    elif view == 'yz':
        hit_data = hit_data[hit_data['h_is_yz']]

    rng = np.random.default_rng(seed)
    n_hits_to_drop = rng.binomial(len(hit_data), 1-hit_survival_prob)
    rprint(f'Dropping {n_hits_to_drop} hits out of {len(hit_data)}, survival probability: {hit_survival_prob*100:0.1f}%, hit that survived: {(1-n_hits_to_drop/len(hit_data))*100:0.1f}%')
    indices_to_drop = rng.choice(hit_data.index, n_hits_to_drop, replace=False)
    hit_data.drop(indices_to_drop, inplace=True)

    rprint('hit_data')
    rprint(hit_data)
    rprint('truth_data')
    rprint(truth_data)

    hit_x_key = "h_pos_z"
    hit_y_key = "h_pos_x" if view == 'xz' else 'h_pos_y'

    binning_x = np.arange(-7477.5, 7577.5, 30) if view == 'xz' else np.arange(-7477.5+15, 7577.5+15, 30)
    binning_y = np.arange(-2625,   2625,   15) if view == 'xz' else np.arange(-2640,      2640,      15)

    hit_x = np.array(hit_data[hit_x_key])
    hit_y = np.array(hit_data[hit_y_key])
    hit_t = np.array(hit_data['h_time'])

    rprint(f'''{hit_x.shape=}
{hit_y.shape=}
{hit_t.shape=}
''')

    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    time_binning = np.arange(0.,np.max(hit_data['h_time']), hit_lifetime_ns)

    if run_clustering:
        from dbscan import DBScan
        hit_counts, xedges, yedges = np.histogram2d(hit_x, hit_y, bins=(binning_x, binning_y))

        x_indices = np.where(hit_counts>0)[0]
        y_indices = np.where(hit_counts>0)[1]
        clusterable_x = (xedges[x_indices]+xedges[x_indices+1])/2
        clusterable_y = (yedges[y_indices]+yedges[y_indices+1])/2

        dbscan = DBScan(
            eps=db_scan_params['eps'],
            min_hits=db_scan_params['min_samples'],
            hit_x=clusterable_x,
            hit_y=clusterable_y
        )
        dbscan.run()
        space_clusters = dbscan.clusters

        print(f'Number of space clusters before prunning: {len(space_clusters)}')
        space_clusters = [cluster for cluster in space_clusters if cluster.n_hits>10]
        print(f'Number of space clusters after prunning: {len(space_clusters)}')

        overall_index = 0

        from time_cluster_maker import Cluster
        clusters = {}

        for i_space, space_cluster in enumerate(space_clusters):

            x_min, x_max, y_min, y_max = space_cluster.get_min_max()
            rprint(f"\n\n\nSpace cluster #{i_space} {x_min=} {x_max=} {y_min=} {y_max=} {space_cluster.n_hits=}")

            hit_t_ = np.array(hit_t[np.where((hit_x>x_min)&(hit_x<x_max)&(hit_y>y_min)&(hit_y<y_max))[0]])

            time_counts, _ = np.histogram(hit_t_, bins=time_binning)

            from time_cluster_maker import TimeClusterMaker
            time_cluster_maker = TimeClusterMaker(time_binning, time_counts, hit_lifetime_ns)
            time_cluster_maker.run_edge_detector()
            time_clusters = time_cluster_maker.clusters


            print(f'Number of time clusters before prunning: {len(time_clusters)}')
            time_clusters = [cluster for cluster in time_clusters if cluster.n_hits>10]
            print(f'Number of time clusters after prunning: {len(time_clusters)}')

            for i_time, time_cluster in enumerate(time_clusters):
                t_min = time_cluster.start
                t_max = time_cluster.stop
                clusters[alphabet[overall_index]] = Cluster.get_from_data(x_min, x_max, y_min, y_max, t_min, t_max, hit_x, hit_y, hit_t)

                def space_reduce(cluster, eps, min_hits):
                    from dbscan import DBScan
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
                    eps=db_scan_params['eps'],
                    min_hits=db_scan_params['min_samples']
                )



                rprint(f"""Time cluster {i_time} & space cluster {i_space} created cluster {overall_index}
    {clusters[alphabet[overall_index]]}
    """)
                overall_index += 1

        for merge in merge_cluster:
            clusters_to_merge = [cluster for label, cluster in clusters.items() if label in merge]
            new_label = merge[0]
            new_cluster = Cluster.union(clusters_to_merge)
            for label in merge:
                del clusters[label]
            clusters[new_label] = new_cluster

        n_decay_clusters = len(clusters) - 1
        rprint(f'after merging clustering: {n_decay_clusters=}')

    else:
        from time_cluster_maker import Cluster
        clusters = {
            "a": Cluster.get_from_data(-5000, 5000, -5000, 5000, 0, 1000, hit_x, hit_y, hit_t),
        }


    fig = plt.figure(figsize=(10,8))

    n_decay_clusters = len(clusters) - 1

    nrows = 3
    ncols_clusters = int(np.ceil(n_decay_clusters/(nrows-1)))
    ncols = 3+ncols_clusters
    empty_rows = n_decay_clusters%(nrows-1)
    rprint(f'{nrows=} {ncols=} {empty_rows=}')

    for cluster in clusters.values():
        rprint(cluster)
    #exit(0)

    gs = GridSpec(nrows=nrows, ncols=ncols, figure=fig, hspace=0.05, wspace=0.05)
    gs.update(left=0.07, right=0.99, bottom=0.07, top=0.99)

    axtime = fig.add_subplot(gs[nrows-1, :])
    axfull = fig.add_subplot(gs[:nrows-1, :3])
    axclus = []


    for i_cluster in range(n_decay_clusters):
        the_col = 3+i_cluster%(ncols_clusters)
        the_row = int(np.floor(i_cluster/(ncols_clusters)))

        ax = fig.add_subplot(gs[the_row, the_col])
        axclus += [ax]




    from rich.table import Table

    cluster_table = Table(
        'Cluster number',
        'Min time',
        'Max time',
        'Min X',
        'Max X',
        'Min Y',
        'Max Y',
        "Number of hits",
        title="Clusters"
    )

    masks = {}
    first_cluster = False
    count1 = 0
    count2 = 3
    count_total = 0

    regions = {}
    time_annotation = {}

    clusters = dict(sorted(clusters.items()))
    if cluster_reindex:
        time_ordered_cluster = {}
        start_times = [cluster.t_min for cluster in clusters.values()]
        labels = list(clusters.keys())
        start_time_sorted_labels = [label for _, label in sorted(zip(start_times, labels))]
        for i, label in enumerate(start_time_sorted_labels):
            time_ordered_cluster[alphabet[i]] = clusters[label]
        clusters = time_ordered_cluster

    n_hits = [cluster.n_hits for cluster in clusters.values()]
    labels = list(clusters.keys())
    hit_sorted_labels = [label for _, label in sorted(zip(n_hits, labels))]

    rprint(f"histogram plotting order: {hit_sorted_labels}")

    axtime.hist(
        [clusters[label].hit_t for label in hit_sorted_labels],
        time_binning,
        log=True,
        histtype='bar',
        stacked=True,
        rasterized=True,
        label=[f'Cluster {label}' for label in hit_sorted_labels],
        #color=CB_color_cycle[:len(hit_sorted_labels)]
    )

    time_counts = np.zeros(len(time_binning)-1)
    for cluster in clusters.values():
        counts, _ = np.histogram(cluster.hit_t, bins=time_binning)
        time_counts += counts

    axtime.set_xlabel('Time [ns]')
    axtime.set_ylabel('Number of hits')
    y_max = np.max(time_counts)*10
    axtime.set_ylim((0.5, y_max))

    handles, labels = axtime.get_legend_handles_labels()
    sorted_labels = sorted(labels)
    sorted_handles = [h for _, h in sorted(zip(labels, handles))]
    axtime.legend(sorted_handles, sorted_labels)

    for label, cluster in clusters.items():

        if count1>3:
            count1 = 0
            count2 += 1

        cluster_table.add_row(
            label,
            str(cluster.t_min), str(cluster.t_max),
            str(cluster.x_min), str(cluster.x_max),
            str(cluster.y_min), str(cluster.y_max),
            str(cluster.n_hits)
        )

        ax = axclus[count_total-1] if count_total>0 else axfull
        # print(f"label = {label}")
        # print(f"title = {cluster_titles.get(label, None)}")
        x_min, x_max, y_min, y_max = plot(
            ax = ax,
            label = label,
            title = cluster_titles.get(label, None),
            hit_data = hit_data,
            hit_key_x = hit_x_key,
            hit_key_y = hit_y_key,
            bin_x = binning_x,
            bin_y = binning_y,
            truth_data = truth_data,
            label_x = None,
            label_y = None,
            time_start = cluster.t_min,
            time_end = cluster.t_max,
            x_min = cluster.x_min,
            x_max = cluster.x_max,
            y_min = cluster.y_min,
            y_max = cluster.y_max,
            with_underlay = count_total>0,
            aspect_ratio = 1,
            plot_axes = plot_axes,
            plot_primaries = plot_primaries.get(label, False),
            plot_particle_ids = plot_particle_ids.get(label, []),
            #plot_list = plot_list.get(label, None),
            legend_label = f"({label})" if count_total>0 else None,
        )


        if count_total>0:
            from matplotlib.patches import Rectangle
            regions[label] = Rectangle([x_min, y_min], width=x_max-x_min, height=y_max-y_min)

        count1 += 1
        count_total += 1

    for ax in fig.get_axes():
        pass

    for label, rect in regions.items():
        axfull.add_patch(rect)
        rect.set(facecolor='none', edgecolor='black', zorder=8)
        position_annotations = [rect.get_x()+100, rect.get_y()+rect.get_height()-150]
        axfull.annotate(label, position_annotations)
        x_min = rect.get_x()
        x_max = rect.get_x()+rect.get_width()
        y_min = rect.get_y()
        y_max = rect.get_y()+rect.get_height()

        x_min_full, x_max_full = axfull.get_xlim()
        y_min_full, y_max_full = axfull.get_ylim()
        padding_x = []
        padding_y = []
        stradling_out_of_bounds = False

        if x_min < x_min_full and x_max > x_min_full:
            padding_x += ["left"]
            x_min_full = x_min
            stradling_out_of_bounds = True
        if x_max > x_max_full and x_min < x_max_full:
            padding_x += ["right"]
            x_max_full = x_max
            stradling_out_of_bounds = True
        if y_min < y_min_full and y_max > y_min_full:
            padding_y += ["left"]
            y_min_full = y_min
            stradling_out_of_bounds = True
        if y_max > y_max_full and y_min < y_max_full:
            padding_y += ["right"]
            y_max_full = y_max
            stradling_out_of_bounds = True

        if stradling_out_of_bounds:

            for px in padding_x:
                if px == "left":
                    x_min_full = add_padding(x_min_full, x_max_full, side=px, padding=5.)
                else:
                    x_max_full = add_padding(x_min_full, x_max_full, side=px, padding=5.)

            for py in padding_y:
                if py == "left":
                    y_min_full = add_padding(y_min_full, y_max_full, side=py, padding=5.)
                else:
                    y_max_full = add_padding(y_min_full, y_max_full, side=py, padding=5.)

            x_min_padded, x_max_padded, y_min_padded, y_max_padded = adjust_for_aspect_ratio(
                x_min_full,
                x_max_full,
                y_min_full,
                y_max_full,
                1
            )
            axfull.set_xlim((x_min_padded, x_max_padded))
            axfull.set_ylim((y_min_padded, y_max_padded))

    for label, rect in regions.items():
        x_min = rect.get_x()
        x_max = rect.get_x()+rect.get_width()
        y_min = rect.get_y()
        y_max = rect.get_y()+rect.get_height()
        x_min_full, x_max_full = axfull.get_xlim()
        y_min_full, y_max_full = axfull.get_ylim()

        if (x_min < x_min_full and x_max < x_min_full or
            x_min > x_max_full and x_max > x_max_full or
            y_min < y_min_full and y_max < y_min_full or
            y_min > y_max_full and y_max > y_max_full):
            rprint(f"{x_min=} {x_max=} {y_min=} {y_max=} {x_min_full=} {x_max_full=} {y_min_full=} {y_max_full=}")
            vector = np.array([
                (x_max+x_min)/2 - (x_max_full+x_min_full)/2,
                (y_max+y_min)/2 - (y_max_full+y_min_full)/2,
            ])
            center = np.array([
                (x_max_full+x_min_full)/2,
                (y_max_full+y_min_full)/2,
            ])
            rprint(f'vector: {vector}')
            unit_vector = vector / np.linalg.norm(vector)
            half_width = (x_max_full - x_min_full) / 2
            arrow_position = center + unit_vector * half_width * 0.9
            arrow_dxdy = unit_vector*half_width*0.08

            axfull.arrow(
                arrow_position[0], arrow_position[1],
                arrow_dxdy[0], arrow_dxdy[1],
                linewidth=2,
                head_width=50,
                fc ='black', ec ='black',
            )
            distance_vector = np.array([
                (x_max-x_min)/2 - arrow_position[0],
                (y_max-y_min)/2 - arrow_position[1],
            ])
            how_far = np.linalg.norm(distance_vector)
            axfull.annotate(
                f"{label}: {how_far/1000:.1f} m",
                arrow_position-np.array([half_width*0.05, half_width*0.05]),
                )


    handles, labels = axfull.get_legend_handles_labels()
    for ax in axclus:
        handles_insert, labels_insert = ax.get_legend_handles_labels()
        handles += handles_insert
        labels += labels_insert
    axfull.legend(handles, labels,ncol=2).set_zorder(20)

    fig.savefig(output, dpi=400)
    rprint(cluster_table)

if __name__ == "__main__":
    main()

