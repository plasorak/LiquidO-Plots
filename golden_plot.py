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


@click.command()
@click.argument('input_data', type=click.Path(exists=True))
@click.argument('output_pdf', type=click.Path(exists=False))
def main(input_data, output_pdf):
    
    hit_lifetime_ns = 5.

    hit_data = uproot.open(input_data)['op_hits'].arrays()
    
    
    hit_x     = ak.to_numpy(hit_data['h_pos_x' ])
    hit_y     = ak.to_numpy(hit_data['h_pos_y' ])
    hit_z     = ak.to_numpy(hit_data['h_pos_z' ])
    hit_t     = ak.to_numpy(hit_data['h_time'  ])
    hit_is_xy = ak.to_numpy(hit_data['h_is_xy' ])
    hit_is_xz = ak.to_numpy(hit_data['h_is_xz' ])
    hit_is_yz = ak.to_numpy(hit_data['h_is_yz' ])

    hit_x_xz = hit_x[hit_is_xz]
    hit_y_xz = hit_y[hit_is_xz]
    hit_z_xz = hit_z[hit_is_xz]
    hit_t_xz = hit_t[hit_is_xz]
    
    hit_x_yz = hit_x[hit_is_yz]
    hit_y_yz = hit_y[hit_is_yz]
    hit_z_yz = hit_z[hit_is_yz]
    hit_t_yz = hit_t[hit_is_yz]
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(output_pdf) as pdf:
            
        fig = plt.figure(figsize=(10,10))
        ax0, ax1, ax2 = fig.subplots(nrows=3)
        
        time_binning = np.arange(0.,np.max(hit_t),hit_lifetime_ns)
        time_counts, _ = np.histogram(hit_t, bins=time_binning)
        ax2.hist(time_binning[:-1], time_binning, weights=time_counts, log=True, rasterized=True)
        ax2.set_xlabel('Time [ns]')
        ax2.set_ylabel('Number of hits')
    
        binning_x_yz = np.arange(-2625,   2625,   15)
        binning_z_yz = np.arange(-7477.5, 7577.5, 30)
        hit_counts_yz, xedges, yedges = np.histogram2d(hit_z_yz, hit_x_yz, bins=(binning_z_yz, binning_x_yz))
        hit_counts_yz = hit_counts_yz.T
        X, Y = np.meshgrid(xedges, yedges)
        i = ax0.pcolormesh(X, Y, hit_counts_yz, norm=colors.LogNorm(), rasterized=True)
        ax0.set_xlabel('z [mm]')
        ax0.set_ylabel('x [mm]')
        ax0.figure.colorbar(i, ax=ax0)
        
    
        binning_y_xz = np.arange(-2640,      2640,      15)
        binning_z_xz = np.arange(-7477.5+15, 7577.5+15, 30)
        hit_counts_xz, xedges, yedges = np.histogram2d(hit_z_xz, hit_y_xz, bins=(binning_z_xz, binning_y_xz))
        hit_counts_xz = hit_counts_xz.T
        X, Y = np.meshgrid(xedges, yedges)
        i = ax1.pcolormesh(X, Y, hit_counts_xz, norm=colors.LogNorm(), rasterized=True)
        ax1.set_xlabel('z [mm]')
        ax1.set_ylabel('y [mm]')
        ax1.figure.colorbar(i, ax=ax1)
        
        fig.tight_layout()
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
        
        for i, cluster in enumerate(clusters):
            cluster_table.add_row(str(i), str(cluster.start), str(cluster.stop), str(cluster.n_hits))
    
            if cluster.n_hits<100:
                continue
            
            mask_xz_past   = hit_t_xz > cluster.start
            mask_xz_future = hit_t_xz < cluster.stop
            mask_xz = mask_xz_future * mask_xz_past
            
            mask_yz_past   = hit_t_yz > cluster.start
            mask_yz_future = hit_t_yz < cluster.stop
            mask_yz = mask_yz_future * mask_yz_past
            
            masks[i] = (mask_xz, mask_yz)
            
            ax0, ax1 = fig.subplots(nrows=2)
            fig.suptitle(f'Time interval: {cluster.start:.0f} ns $\\rightarrow$ {cluster.stop:.0f} ns')
            
            hit_counts_yz, xedges, yedges = np.histogram2d(hit_z_yz[mask_yz], hit_x_yz[mask_yz], bins=(binning_z_yz, binning_x_yz))
            hit_counts_yz = hit_counts_yz.T
            X, Y = np.meshgrid(xedges, yedges)
            im = ax0.pcolormesh(X, Y, hit_counts_yz, norm=colors.LogNorm(), rasterized=True)
            ax0.set_xlabel('z [mm]')
            ax0.set_ylabel('x [mm]')
            ax0.figure.colorbar(im, ax=ax0)
            
            hit_counts_xz, xedges, yedges = np.histogram2d(hit_z_xz[mask_xz], hit_y_xz[mask_xz], bins=(binning_z_xz, binning_y_xz))
            hit_counts_xz = hit_counts_xz.T
            X, Y = np.meshgrid(xedges, yedges)
            im = ax1.pcolormesh(X, Y, hit_counts_xz, norm=colors.LogNorm(), rasterized=True)
            ax1.set_xlabel('z [mm]')
            ax1.set_ylabel('y [mm]')
            ax1.figure.colorbar(im, ax=ax1)
    
            fig.tight_layout()
            pdf.savefig()
            fig.clf()
            
            
        rprint(cluster_table)
    
if __name__ == "__main__":
    main()
    
