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


np.set_printoptions(threshold=sys.maxsize)


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
    print(f'Aspect ratio: {current_aspect_ratio} -> {new_aspect_ratio}')
    return x_min, x_max, y_min, y_max


def get_norm(x_min, x_max, y_min, y_max):
    return np.sqrt((x_max - x_min)*(x_max - x_min) + (y_max - y_min)*(y_max - y_min))

def add_truth_particle(ax, hit_data, hit_key_x, hit_key_y, truth_data, x_min, x_max, y_min, y_max, time_start, time_end):

    global particle_color

    truth_key_x = hit_key_x.replace('h_pos_', 'i_pos_')
    truth_key_y = hit_key_y.replace('h_pos_', 'i_pos_')

    local_truth_data = truth_data.where(truth_data[truth_key_x]>x_min)
    local_truth_data.where(truth_data[truth_key_x]<x_max, inplace=True)
    local_truth_data.where(truth_data[truth_key_y]>y_min, inplace=True)
    local_truth_data.where(truth_data[truth_key_y]<y_max, inplace=True)

    all_track_ids = np.unique(local_truth_data['track_id'])

    plotted_track_ids = []
    primary = []

    n_hits = len(hit_data)

    for id in all_track_ids:
        track_data = local_truth_data[local_truth_data['track_id'] == id]
        if (track_data['parent_id'] == 0).any():
            plotted_track_ids += [id]
            primary += [id]
            continue

        particle_hits = hit_data[hit_data['h_parent_id']==id]
        this_n_hits = particle_hits.shape[0]
        if this_n_hits>0.05*n_hits and this_n_hits>10:
            rprint(f"Track {id} has {this_n_hits} hits, plotting")
            plotted_track_ids += [id]



    for id in plotted_track_ids:

        # mask1 = part_id<track_id+10
        mask = local_truth_data['track_id'] == id
        this_track = local_truth_data[mask]
        xs = this_track[truth_key_x]
        ys = this_track[truth_key_y]
        pdg = this_track.iloc[0]['i_particle']
        if id in primary:
            xs = pd.concat([pd.Series([0]), xs])
            ys = pd.concat([pd.Series([0]), ys])
        p = Particle.from_pdgid(pdg)
        e = np.max(this_track['i_E'])
        M = p.mass
        if M is None and abs(pdg) in [12, 14, 16]:
            M = 0

        ke = e - M
        rprint(f"{pdg=} {M=} {ke=} {e=} {p.name=}")


        primary_str = ' (primary)' if id in primary else ''
        rprint(f"Plotting {pdg}{primary_str}")

        norm = get_norm(
            np.min(xs), np.max(xs),
            np.min(ys), np.max(ys),
        )
        diag = get_norm(x_min, x_max, y_min, y_max)

        if norm > 0.01*diag:
            label = f'${p.latex_name}$ KE {ke:.1f} MeV{primary_str}' if p is not None else f'{pdg} E {e} MeV'

            if id in particle_color:
                label = None

            lines = ax.plot(
                xs,
                ys,
                linewidth=2,
                label = label,
                color = particle_color[id] if id in particle_color else None,
            )
            particle_color[id] = lines[0].get_c()
        elif id in primary:
            #x_min, x_max, y_min, y_max

            xs = xs.array
            ys = ys.array
            x0 = xs[0]
            y0 = ys[0]
            dx = (xs[1]-xs[0])
            dy = (ys[1]-ys[0])
            norm = get_norm(
                xs[0], xs[1],
                ys[0], ys[1]
            )

            new_norm = 0.05 * diag
            factor = new_norm / norm
            new_dx = dx * factor
            new_dy = dy * factor

            lines = ax.arrow(
                x0,y0,
                new_dx, new_dy,
                linewidth=2,
                head_width=50,
                fc ='black', ec ='black',
                label = f'${p.latex_name}$ KE {ke:.1f} MeV{primary_str}',
                #color = particle_color[id] if id in particle_color else None,
            )
            #particle_color[id] = lines.get_color()



def plot(
    ax,
    label,
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
    with_underlay = True,
    with_legend = False,
    aspect_ratio = 1,
):
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

    rprint(f'Before padding {x_min=} {x_max=} {y_min=} {y_max=}')

    # x_min_padded = add_padding(x_min, x_max, 'left',  padding)
    # x_max_padded = add_padding(x_min, x_max, 'right', padding)
    # y_min_padded = add_padding(y_min, y_max, 'left',  padding)
    # y_max_padded = add_padding(y_min, y_max, 'right', padding)

    x_min_padded, x_max_padded, y_min_padded, y_max_padded = adjust_for_aspect_ratio(x_min, x_max, y_min, y_max, aspect_ratio)

    x_min = x_min_padded
    x_max = x_max_padded
    y_min = y_min_padded
    y_max = y_max_padded

    rprint(f'After padding {x_min=} {x_max=} {y_min=} {y_max=}')

    if with_underlay:
        add_truth_particle(ax, hit_data_past, hit_key_x, hit_key_y, truth_data, x_min, x_max, y_min, y_max, 0, time_end)

    add_truth_particle(ax, hit_data_present, hit_key_x, hit_key_y, truth_data, x_min, x_max, y_min, y_max, time_start, time_end)

    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))

    if label is not None:

        ax.annotate(
            label,
            (0.05, 0.95),# if not with_underlay else (0.05, 0.95),
            xycoords='axes fraction',
            va='center'
        )

    if label_x is not None:
        ax.set_xlabel(f'{label_x} [mm]')

    if label_y is not None:
        ax.set_ylabel(f'{label_y} [mm]')

    if with_legend:
        ax.legend().set_zorder(20)

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

def merge_callback(ctx, param, value):
    return value.split(',')

@click.command()
@click.argument('input_data', type=click.Path(exists=True))
@click.argument('output', type=click.Path(exists=False))
@click.option('--hit-threshold', type=int, default=None, help='Plot truth data from the tracks that have at least this number of hits')
@click.option('--view', type=str, default='xz')
@click.option('--no-reindex', is_flag=True, default=False)
@click.option('--highest-hit-contributors', type=int, default=None, help='Plot truth data from the number of tracks that contribute to the most number of hits')
@click.option('--merge-clusters', type=str, default="a", help='a list of list of clusters to merge: format = "abc,def" to merge a, b and c together, and d, e and f together', callback=merge_callback)
def main(input_data, output, hit_threshold, view, no_reindex, highest_hit_contributors, merge_clusters):

    if hit_threshold is None and highest_hit_contributors is None:
        hit_threshold = 1000

    hit_lifetime_ns = 5.

    hit_data   = uproot.open(input_data)['op_hits'] .arrays(library="pd")
    truth_data = uproot.open(input_data)['mc_truth'].arrays(library="pd")
    if   view == 'xz':
        hit_data = hit_data[hit_data['h_is_xz']]
    elif view == 'yz':
        hit_data = hit_data[hit_data['h_is_yz']]

    rprint('hit_data')
    rprint(hit_data)
    rprint('truth_data')
    rprint(truth_data)

    hit_x_key = "h_pos_z"
    hit_y_key = "h_pos_x" if view == 'xz' else 'h_pos_y'

    hit_x = hit_data[hit_x_key]
    hit_y = hit_data[hit_y_key]

    time_binning = np.arange(0.,np.max(hit_data['h_time']), hit_lifetime_ns)
    time_counts, _ = np.histogram(hit_data['h_time'], bins=time_binning)

    from time_cluster_maker import TimeClusterMaker
    time_cluster_maker = TimeClusterMaker(time_binning, time_counts, hit_lifetime_ns)
    time_cluster_maker.run_edge_detector()
    time_clusters = time_cluster_maker.clusters
    # Manually make the first cluster include all the hit
    time_clusters[0].stop = time_clusters[-1].stop

    n_decay_clusters = len(time_clusters) - 1
    rprint(f'before removing: {n_decay_clusters=}')
    time_clusters = [cluster for cluster in time_clusters if cluster.n_hits>100]
    n_decay_clusters = len(time_clusters) - 1
    rprint(f'after removing: {n_decay_clusters=}')


    from time_cluster_maker import Cluster
    clusters = {}


    from dbscan import DBScan

    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    overall_index = 1

    for i, time_cluster in enumerate(time_clusters):

        if i == 0:
            clusters[alphabet[0]] = Cluster.get_from_time_and_space_clusters(time_cluster, None)
            continue

        mask_past   = hit_data["h_time"]>time_cluster.start
        mask_future = hit_data["h_time"]<time_cluster.stop
        mask = mask_past * mask_future

        hit_x_ = np.array(hit_x[mask].values)
        hit_y_ = np.array(hit_y[mask].values)

        dbscan = DBScan(eps=500, min_hits=5, hit_x=hit_x_, hit_y=hit_y_)
        dbscan.run()
        for j, space_cluster in enumerate(dbscan.clusters):
            clusters[alphabet[overall_index]] = Cluster.get_from_time_and_space_clusters(time_clusters[i], space_cluster)
            rprint(f"""Time cluster {i} & space cluster {j} created cluster {overall_index}
{clusters[alphabet[overall_index]]}
""")
            overall_index += 1

    n_decay_clusters = len(clusters) - 1
    rprint(f'after space clustering: {n_decay_clusters=}')

    for merge in merge_clusters:
        clusters_to_merge = [cluster for label, cluster in clusters.items() if label in merge]
        new_label = merge[0]
        new_cluster = Cluster.union(clusters_to_merge)
        for label in merge:
            del clusters[label]
        clusters[new_label] = new_cluster

    n_decay_clusters = len(clusters) - 1
    rprint(f'after merging clustering: {n_decay_clusters=}')

    fig = plt.figure(figsize=(10,8))

    n_decay_clusters = len(clusters) - 1

    nrows = 3
    ncols_clusters = int(np.ceil(n_decay_clusters/(nrows-1)))
    ncols = 3+ncols_clusters
    empty_rows = n_decay_clusters%(nrows-1)
    print(f'{nrows=} {ncols=} {empty_rows=}')

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


    axtime.hist(time_binning[:-1], time_binning, weights=time_counts, log=True, rasterized=True)
    axtime.set_xlabel('Time [ns]')
    axtime.set_ylabel('Number of hits')
    y_max = np.max(time_counts)*10
    axtime.set_ylim((0.5, y_max))


    binning_x = np.arange(-7477.5, 7577.5, 30) if view == 'xz' else np.arange(-7477.5+15, 7577.5+15, 30)
    binning_y = np.arange(-2625,   2625,   15) if view == 'xz' else np.arange(-2640,      2640,      15)

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
    if not no_reindex:
        new_clusters = {}
        for i, cluster in enumerate(clusters.values()):
            label = alphabet[i-1] if i>0 else 'all'
            new_clusters[label] = cluster

        clusters = new_clusters
    else:
        clusters['all (a)'] = clusters.pop('a')
        clusters = dict(sorted(clusters.items()))


    for label, cluster in clusters.items():

        if count1>3:
            count1 = 0
            count2 += 1

        cluster_table.add_row(
            label,
            str(cluster.time_start), str(cluster.time_stop),
            str(cluster.x_min), str(cluster.x_max),
            str(cluster.y_min), str(cluster.y_max),
            str(cluster.n_hits)
        )

        padding = 0

        axtime.annotate(
            label,
            xy=(cluster.time_start, cluster.n_hits*3),
            xycoords='data'
        )

        ax = axclus[count_total-1] if count_total>0 else axfull

        x_min, x_max, y_min, y_max = plot(
            ax = ax,
            label = label,
            hit_data = hit_data,
            hit_key_x = hit_x_key,
            hit_key_y = hit_y_key,
            bin_x = binning_x,
            bin_y = binning_y,
            truth_data = truth_data,
            label_x = None,
            label_y = None,
            time_start = cluster.time_start,
            time_end = cluster.time_stop,
            x_min = cluster.x_min,
            x_max = cluster.x_max,
            y_min = cluster.y_min,
            y_max = cluster.y_max,
            with_legend = count_total==0,
            with_underlay = count_total>0,
            aspect_ratio = 1,
        )


        if count_total>0:
            from matplotlib.patches import Rectangle
            regions[label] = Rectangle([x_min, y_min], width=x_max-x_min, height=y_max-y_min)

        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

        count1 += 1
        count_total += 1

    for ax in fig.get_axes():
        pass

    for label, rect in regions.items():
        axfull.add_patch(rect)
        rect.set(facecolor='none', edgecolor='black', zorder=8)
        position_annotations = [rect.get_x()+100, rect.get_y()+rect.get_height()-150]
        axfull.annotate(label, position_annotations)


    fig.savefig(output, dpi=400)
    rprint(cluster_table)

if __name__ == "__main__":
    main()

