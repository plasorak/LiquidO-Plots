import awkward as ak
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import uproot
from rich import print as rprint
import matplotlib.animation as animation
import sys
from particle import Particle
import click
import pandas as pd
np.set_printoptions(threshold=sys.maxsize)

def add_padding(left, right, side:str, padding:float=10.):
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

    # print(f'{number=}')
    return number


def add_truth_particle(ax, hit_track, part_x, part_y, part_id, part_pdg, part_e, hit_threshold, highest_hit_contributors):
    all_hit_track_ids = np.unique(hit_track)

    how_many_hits_per_track_id = {}
    for hit_track_id in all_hit_track_ids:
        how_many_hits_per_track_id[hit_track_id] = np.count_nonzero(hit_track==hit_track_id)

    if highest_hit_contributors is not None:
        number_of_hits = list(how_many_hits_per_track_id.values())
        number_of_hits.sort()
        number_of_hits = number_of_hits[::-1]
        if highest_hit_contributors >= len(number_of_hits):
            hit_threshold = 0
        else:
            hit_threshold_ = number_of_hits[highest_hit_contributors-1]-1
            if hit_threshold is not None and hit_threshold_ > hit_threshold:
                hit_threshold = hit_threshold_
            if hit_threshold is None:
                hit_threshold = hit_threshold_



    for track_id, nhit in how_many_hits_per_track_id.items():
        if nhit<hit_threshold:
            continue

        # mask1 = part_id<track_id+10
        mask = part_id==track_id
        # mask = mask1 * mask2
        if np.count_nonzero(mask) == 0:
            print(f'{track_id=} of {nhit=} seems to have no truth information!')
            continue

        xs = part_x[mask]
        ys = part_y[mask]
        pdg = part_pdg[mask]
        pdg = pdg[0] if len(pdg)>0 else None
        p = Particle.from_pdgid(pdg)
        e = np.max(part_e[mask])
        M = p.mass
        ke = e - M
        if ke<10:
            continue
        ax.plot(xs, ys, linewidth=2, label=f'${p.latex_name}$ KE {ke:.1f} MeV')


def plot(
        fig,
        hit_x, hit_y, hit_id,
        bin_x, bin_y,
        hit_threshold, highest_hit_contributors,
        truth_x, truth_y, truth_id, truth_pdg, truth_e,
        label_x, label_y,
        underlay_x=None, underlay_y=None, underlay_id=None,
        padding = 10
    ):

    ax = fig.subplots(nrows=1)

    hit_counts, xedges, yedges = np.histogram2d(hit_x, hit_y, bins=(bin_x, bin_y))
    hit_counts = hit_counts.T

    if underlay_x is not None:
        underlay_hit_counts, xedges, yedges = np.histogram2d(underlay_x, underlay_y, bins=(bin_x, bin_y))
        underlay_hit_counts = underlay_hit_counts.T

    X, Y = np.meshgrid(xedges, yedges)

    underlay_colormesh = None
    if underlay_x is not None:
        underlay_hit_counts = np.ma.masked_array(underlay_hit_counts, underlay_hit_counts<0.5)
        underlay_colormesh = ax.pcolormesh(X, Y, underlay_hit_counts, norm='log', rasterized=True, cmap='Greys', vmin=np.min(underlay_hit_counts), vmax=np.max(underlay_hit_counts)*10)

    #hit_counts = hit_counts[hit_counts>0.5]
    hit_counts = np.ma.masked_array(hit_counts, hit_counts<0.5)
    i = ax.pcolormesh(X, Y, hit_counts, norm='log', rasterized=True, vmin=np.min(hit_counts), vmax=np.max(hit_counts))

    x_min = add_padding(np.min(hit_x), np.max(hit_x), 'left',  padding)
    x_max = add_padding(np.min(hit_x), np.max(hit_x), 'right', padding)

    y_min = add_padding(np.min(hit_y), np.max(hit_y), 'left',  padding)
    y_max = add_padding(np.min(hit_y), np.max(hit_y), 'right', padding)

    if underlay_id is not None:
        mask_low_x  = underlay_x>x_min
        mask_high_x = underlay_x<x_max
        mask_low_y  = underlay_y>y_min
        mask_high_y = underlay_y<y_max
        mask = mask_low_x*mask_high_x*mask_low_y*mask_high_y
        add_truth_particle(ax, underlay_id[mask], truth_x, truth_y, truth_id, truth_pdg, truth_e, 500, None)

    add_truth_particle(ax, hit_id, truth_x, truth_y, truth_id, truth_pdg, truth_e, hit_threshold, highest_hit_contributors)

    ax.set_xlabel(f'{label_x} [mm]')
    ax.set_ylabel(f'{label_y} [mm]')
    ax.legend()
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))

    if underlay_colormesh:
        ax.figure.colorbar(underlay_colormesh, ax=ax)

    ax.figure.colorbar(i, ax=ax)
    fig.tight_layout()


@click.command()
@click.argument('input_data', type=click.Path(exists=True))
@click.argument('output_pdf', type=click.Path(exists=False))
@click.option('--hit-threshold', type=int, default=None, help='Plot truth data from the tracks that have at least this number of hits')
@click.option('--highest-hit-contributors', type=int, default=None, help='Plot truth data from the number of tracks that contribute to the most number of hits')
def main(input_data, output_pdf, hit_threshold, highest_hit_contributors):

    if hit_threshold is None and highest_hit_contributors is None:
        hit_threshold = 1000

    hit_lifetime_ns = 5.

    hit_data   = uproot.open(input_data)['op_hits'].arrays()
    truth_data = uproot.open(input_data)['mc_truth'].arrays()

    hit_x     = ak.to_numpy(hit_data['h_pos_x'     ])
    hit_y     = ak.to_numpy(hit_data['h_pos_y'     ])
    hit_z     = ak.to_numpy(hit_data['h_pos_z'     ])
    hit_t     = ak.to_numpy(hit_data['h_time'      ])
    hit_is_xy = ak.to_numpy(hit_data['h_is_xy'     ])
    hit_is_xz = ak.to_numpy(hit_data['h_is_xz'     ])
    hit_is_yz = ak.to_numpy(hit_data['h_is_yz'     ])
    hit_track = ak.to_numpy(hit_data['h_parent_id'])

    hit_x_xz  = hit_x    [hit_is_xz]
    hit_y_xz  = hit_y    [hit_is_xz]
    hit_z_xz  = hit_z    [hit_is_xz]
    hit_t_xz  = hit_t    [hit_is_xz]
    hit_id_xz = hit_track[hit_is_xz]

    hit_x_yz  = hit_x    [hit_is_yz]
    hit_y_yz  = hit_y    [hit_is_yz]
    hit_z_yz  = hit_z    [hit_is_yz]
    hit_t_yz  = hit_t    [hit_is_yz]
    hit_id_yz = hit_track[hit_is_yz]

    truth_id        = ak.to_numpy(truth_data['track_id'  ])
    truth_parent_id = ak.to_numpy(truth_data['parent_id' ])
    truth_pdg       = ak.to_numpy(truth_data['i_particle'])
    truth_x         = ak.to_numpy(truth_data['i_pos_x'   ])
    truth_y         = ak.to_numpy(truth_data['i_pos_y'   ])
    truth_z         = ak.to_numpy(truth_data['i_pos_z'   ])
    truth_t         = ak.to_numpy(truth_data['i_time'    ])
    truth_E         = ak.to_numpy(truth_data['i_E'       ])



    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(output_pdf) as pdf:

        fig = plt.figure(figsize=(10,8))
        ax = fig.subplots(nrows=1)

        time_binning = np.arange(0.,np.max(hit_t),hit_lifetime_ns)
        time_counts, _ = np.histogram(hit_t, bins=time_binning)
        ax.hist(time_binning[:-1], time_binning, weights=time_counts, log=True, rasterized=True)
        ax.set_xlabel('Time [ns]')
        ax.set_ylabel('Number of hits')
        pdf.savefig()
        fig.clf()


        fig.suptitle(f'Full event, ZX view')
        binning_x_yz = np.arange(-2625,   2625,   15)
        binning_z_yz = np.arange(-7477.5, 7577.5, 30)
        plot(
            fig,
            hit_x = hit_z_yz,
            hit_y = hit_x_yz,
            hit_id = hit_id_yz,
            bin_x = binning_z_yz,
            bin_y = binning_x_yz,
            hit_threshold = hit_threshold,
            highest_hit_contributors = highest_hit_contributors,
            truth_x = truth_z,
            truth_y = truth_x,
            truth_id = truth_id,
            truth_pdg = truth_pdg,
            truth_e = truth_E,
            label_x = 'z',
            label_y = 'x',
        )
        pdf.savefig()
        fig.clf()


        fig.suptitle(f'Full event, ZY view')
        binning_y_xz = np.arange(-2640,      2640,      15)
        binning_z_xz = np.arange(-7477.5+15, 7577.5+15, 30)
        plot(
            fig,
            hit_x = hit_z_xz,
            hit_y = hit_y_xz,
            hit_id = hit_id_yz,
            bin_x = binning_z_xz,
            bin_y = binning_y_xz,
            hit_threshold = hit_threshold,
            highest_hit_contributors = highest_hit_contributors,
            truth_x = truth_z,
            truth_y = truth_y,
            truth_id = truth_id,
            truth_pdg = truth_pdg,
            truth_e = truth_E,
            label_x = 'z',
            label_y = 'y',
        )
        pdf.savefig()
        fig.clf()


        class Cluster:
            def __init__(self, start):
                self.start = start
                self.stop = start
                self.n_hits = 0

            def grow(self, until, nhits):
                self.stop = until
                self.n_hits += nhits

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

        from rich.table import Table

        cluster_table = Table('Cluster', 'Min time', 'Max time', "Number of hits", title="Time clusters")

        masks = {}
        first_cluster = True

        for i, cluster in enumerate(clusters):
            cluster_table.add_row(str(i), str(cluster.start), str(cluster.stop), str(cluster.n_hits))

            if cluster.n_hits<100:
                continue

            mask_xz_future = hit_t_xz > cluster.start
            mask_xz_past   = hit_t_xz < cluster.stop
            mask_xz = mask_xz_future * mask_xz_past

            mask_yz_future = hit_t_yz > cluster.start
            mask_yz_past   = hit_t_yz < cluster.stop
            mask_yz = mask_yz_future * mask_yz_past

            masks[i] = (mask_xz, mask_yz)

            fig.suptitle(f'Time interval: {cluster.start:.0f} ns $\\rightarrow$ {cluster.stop:.0f} ns, ZX view')
            plot(
                fig,
                hit_x = hit_z_yz[mask_yz],
                hit_y = hit_x_yz[mask_yz],
                hit_id = hit_id_yz[mask_yz],
                bin_x = binning_z_yz,
                bin_y = binning_x_yz,
                hit_threshold = 20,
                highest_hit_contributors = highest_hit_contributors,
                truth_x = truth_z,
                truth_y = truth_x,
                truth_id = truth_id,
                truth_pdg = truth_pdg,
                truth_e = truth_E,
                label_x = 'z',
                label_y = 'x',
                underlay_x = hit_z_yz[mask_yz_past] if not first_cluster else None,
                underlay_y = hit_x_yz[mask_yz_past] if not first_cluster else None,
                underlay_id = hit_id_yz[mask_yz_past] if not first_cluster else None,
                padding = 50 if not first_cluster else 10,
            )
            pdf.savefig()
            fig.clf()

            fig.suptitle(f'Time interval: {cluster.start:.0f} ns $\\rightarrow$ {cluster.stop:.0f} ns, ZY view')
            plot(
                fig,
                hit_x = hit_z_xz[mask_xz],
                hit_y = hit_y_xz[mask_xz],
                hit_id = hit_id_xz[mask_xz],
                bin_x = binning_z_xz,
                bin_y = binning_y_xz,
                hit_threshold = 20,
                highest_hit_contributors = highest_hit_contributors,
                truth_x = truth_z,
                truth_y = truth_y,
                truth_id = truth_id,
                truth_pdg = truth_pdg,
                truth_e = truth_E,
                label_x = 'z',
                label_y = 'x',
                underlay_x = hit_z_xz[mask_xz_past] if not first_cluster else None,
                underlay_y = hit_y_xz[mask_xz_past] if not first_cluster else None,
                underlay_id = hit_id_xz[mask_xz_past] if not first_cluster else None,
                padding = 50 if not first_cluster else 10,
            )
            pdf.savefig()
            fig.clf()

            first_cluster = False

        rprint(cluster_table)

if __name__ == "__main__":
    main()

