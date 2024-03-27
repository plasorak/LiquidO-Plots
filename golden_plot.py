import awkward as ak
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

    for id in all_track_ids:
        track_data = local_truth_data[local_truth_data['track_id'] == id]
        if (track_data['parent_id'] == 0).any():
            plotted_track_ids += [id]
            primary += [id]

        n_hits = np.count_nonzero(hit_data[hit_data['h_track_id']==id])
        if n_hits>100:
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
        ke = e - M

        primary_str = ' (primary)' if id in primary else ''
        rprint(f"Plotting {pdg}{primary_str}")
        rprint(f"{xs=}\n{ys=}")

        if np.max(xs) - np.min(xs) > 100:
            lines = ax.plot(
                xs,
                ys,
                linewidth=2,
                label = f'${p.latex_name}$ KE {ke:.1f} MeV{primary_str}',
                color = particle_color[id] if id in particle_color else None,
            )
            particle_color[id] = lines[0].get_c()
        else:
            #x_min, x_max, y_min, y_max

            xs = xs.array
            ys = ys.array
            x0 = xs[0]
            y0 = ys[0]
            dx = (xs[1]-xs[0])
            dy = (ys[1]-ys[0])

            norm = np.sqrt((dx*dx) + (dy*dy))
            new_norm = 0.05*np.sqrt((x_max - x_min)*(x_max - x_min) + (y_max - y_min)*(y_max - y_min))
            factor =new_norm / norm
            new_dx = dx * factor
            new_dy = dy * factor

            lines = ax.arrow(
                x0,y0,
                new_dx, new_dy,
                linewidth=2,
                head_width=6,
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
    padding = 10,
    with_underlay=True,
    with_legend=False,
):
    hit_data_present = hit_data.where(hit_data["h_time"]>time_start, inplace=False)# * hit_data["h_time"]<time_end
    hit_data_present = hit_data_present.where(hit_data["h_time"]<time_end)# * hit_data["h_time"]<time_end

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

    x_min = add_padding(np.min(hit_data_present[hit_key_x]), np.max(hit_data_present[hit_key_x]), 'left',  padding)
    x_max = add_padding(np.min(hit_data_present[hit_key_x]), np.max(hit_data_present[hit_key_x]), 'right', padding)

    y_min = add_padding(np.min(hit_data_present[hit_key_y]), np.max(hit_data_present[hit_key_y]), 'left',  padding)
    y_max = add_padding(np.min(hit_data_present[hit_key_y]), np.max(hit_data_present[hit_key_y]), 'right', padding)

    if False:#with_underlay:
        add_truth_particle(ax, hit_data_past, hit_key_x, hit_key_y, truth_data, x_min, x_max, y_min, y_max, 0, time_end)

    add_truth_particle(ax, hit_data_present, hit_key_x, hit_key_y, truth_data, x_min, x_max, y_min, y_max, time_start, time_end)

    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))

    if label is not None:

        ax.annotate(
            label,
            (0.025, 0.975) if not with_underlay else (0.05, 0.95),
            xycoords='axes fraction',
            va='center'
        )

    if label_x is not None:
        ax.set_xlabel(f'{label_x} [mm]')

    if label_y is not None:
        ax.set_ylabel(f'{label_y} [mm]')

    if with_legend:
        ax.legend().set_zorder(20)



@click.command()
@click.argument('input_data', type=click.Path(exists=True))
@click.argument('output', type=click.Path(exists=False))
@click.option('--hit-threshold', type=int, default=None, help='Plot truth data from the tracks that have at least this number of hits')
@click.option('--view', type=str, default='xz')
@click.option('--highest-hit-contributors', type=int, default=None, help='Plot truth data from the number of tracks that contribute to the most number of hits')
def main(input_data, output, hit_threshold, view, highest_hit_contributors):

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

    time_binning = np.arange(0.,np.max(hit_data['h_time']), hit_lifetime_ns)
    time_counts, _ = np.histogram(hit_data['h_time'], bins=time_binning)

    class Cluster:
        def __init__(self, start):
            self.start = start
            self.stop = start
            self.n_hits = 0
            self.max_hits = 0

        def grow(self, until, nhits):
            self.stop = until
            self.n_hits += nhits
            if nhits > self.max_hits:
                self.max_hits = nhits

    current_cluster = None
    clusters = []
    from copy import deepcopy as dc

    for ibins, nhit in np.ndenumerate(time_counts):
        if current_cluster is not None: # already growing a cluster
            if nhit>0: # some hits, we continue to grow
                current_cluster.grow(time_binning[ibins]+hit_lifetime_ns, nhit)
            elif nhit==0: # no hits, stop the cluster
                clusters.append(dc(current_cluster))
                current_cluster = None
        else: # not in a cluster
            if nhit>0: # need to start a new cluster
                current_cluster = Cluster(time_binning[ibins])
                current_cluster.grow(time_binning[ibins]+hit_lifetime_ns, nhit)
            elif nhit==0:
                pass

    n_decay_clusters = len(clusters) - 1
    rprint(f'before removing: {n_decay_clusters=}')
    clusters = [cluster for cluster in clusters if cluster.n_hits>100]
    n_decay_clusters = len(clusters) - 1
    rprint(f'after removing: {n_decay_clusters=}')

    n_decay_clusters = len(clusters) - 1
    fig = plt.figure(figsize=(10,8))

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

    hit_x_key = "h_pos_z"
    hit_y_key = "h_pos_x" if view == 'xz' else 'h_pos_y'

    binning_x = np.arange(-7477.5, 7577.5, 30) if view == 'xz' else np.arange(-7477.5+15, 7577.5+15, 30)
    binning_y = np.arange(-2625,   2625,   15) if view == 'xz' else np.arange(-2640,      2640,      15)

    from rich.table import Table

    cluster_table = Table('Cluster', 'Min time', 'Max time', "Number of hits", title="Time clusters")

    masks = {}
    first_cluster = False
    count1 = 0
    count2 = 3
    count_total = 0
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    regions = {}
    time_annotation = {}

    for i, cluster in enumerate(clusters):

        if count1>3:
            count1 = 0
            count2 += 1

        cluster_table.add_row(str(i), str(cluster.start), str(cluster.stop), str(cluster.n_hits))

        padding = 10

        label = f'{alphabet[count_total]})'

        axtime.annotate(
            label,
            xy=(cluster.start, cluster.max_hits*3),
            xycoords='data'
        )

        ax = axclus[i-1] if i>0 else axfull

        plot(
            ax = ax,
            label = None,
            hit_data = hit_data,
            hit_key_x = hit_x_key,
            hit_key_y = hit_y_key,
            bin_x = binning_x,
            bin_y = binning_y,
            truth_data = truth_data,
            label_x = None,
            label_y = None,
            time_start = cluster.start,
            time_end = cluster.stop,
            with_legend = i==0,
            with_underlay = i>0,

        )


        if i>0:
            from matplotlib.patches import Rectangle
            #regions[label] = Rectangle([x_min, y_min], width=x_max-x_min, height=y_max-y_min)

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

