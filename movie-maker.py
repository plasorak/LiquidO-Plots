import awkward as ak
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import uproot
from rich import print as rprint
import matplotlib.animation as animation
import sys
np.set_printoptions(threshold=sys.maxsize)
from particle import Particle
import click
import pandas as pd

# TODO: here
# - Add truth information

@click.command()
@click.argument('input_data', type=click.Path(exists=True))
@click.argument('output_movie', type=click.Path())
@click.option('--only-image', is_flag=True, default=False)
@click.option('--view', type=click.Choice(['xz', 'yz']), default='yz')
@click.option('-e', '--energy-cut', type=float, default=0)
def main(input_data, output_movie, only_image, view, energy_cut):

    hit_lifetime_ns_1 = 2.
    hit_lifetime_ns_2 = 100.
    cut_ns = 200
    print_every_ns = 100.
    fps = 10.

    if output_movie[-4:] != ".mp4":
        raise RuntimeError('output file must be finishing with .mp4')

    hit_data  = uproot.open(input_data)['op_hits' ].arrays()
    geom_data = uproot.open(input_data)['geom'    ].arrays()
    mc_truth  = uproot.open(input_data)['mc_truth'].arrays()

    hit_x     = ak.to_numpy(hit_data['h_pos_x' ])
    hit_y     = ak.to_numpy(hit_data['h_pos_y' ])
    hit_z     = ak.to_numpy(hit_data['h_pos_z' ])
    hit_t     = ak.to_numpy(hit_data['h_time'  ])
    hit_is_xy = ak.to_numpy(hit_data['h_is_xy' ])
    hit_is_xz = ak.to_numpy(hit_data['h_is_xz' ])
    hit_is_yz = ak.to_numpy(hit_data['h_is_yz' ])

    rprint(f'nhits {len(hit_x)}')

    if view == 'z':
        hit_x = hit_x[hit_is_xz]
        hit_y = hit_y[hit_is_xz]
        hit_z = hit_z[hit_is_xz]
        hit_t = hit_t[hit_is_xz]
    elif view == 'xz':
        hit_x = hit_x[hit_is_yz]
        hit_y = hit_y[hit_is_yz]
        hit_z = hit_z[hit_is_yz]
        hit_t = hit_t[hit_is_yz]

    rprint(f'nhits {len(hit_x)}')

    im = plt.hist2d(hit_z, hit_x, bins=[100,100])

    plt.xlabel("z [cm]")
    plt.ylabel("x [cm]")
    # plt.xlim([min_x, max_x])
    # plt.ylim([min_z, max_z])
    plt.colorbar()
    plt.tight_layout()

    plt.savefig(output_movie+'.hits.png', dpi=800)
    #exit()

    particle_x            = ak.to_numpy(mc_truth['i_pos_x'])
    particle_y            = ak.to_numpy(mc_truth['i_pos_y'])
    particle_z            = ak.to_numpy(mc_truth['i_pos_z'])
    particle_t            = ak.to_numpy(mc_truth['i_time' ])
    particle_ids          = ak.to_numpy(mc_truth['track_id'])
    particle_pdgs         = ak.to_numpy(mc_truth['i_particle'])
    particle_energies     = ak.to_numpy(mc_truth['i_E'])
    particle_parents      = ak.to_numpy(mc_truth['parent_id'])
    particle_interactions = ak.to_numpy(mc_truth['interaction_id'])

    # Chuck all these neutrinos that escape the detector
    neutrino_pdgs = [12,-12,14,-14,16,-16]
    neutrino_mask = particle_pdgs != 0
    for neutrino_pdg in neutrino_pdgs:
        neutrino_mask *= particle_pdgs != neutrino_pdg

    particle_x            = particle_x            [neutrino_mask]
    particle_y            = particle_y            [neutrino_mask]
    particle_z            = particle_z            [neutrino_mask]
    particle_t            = particle_t            [neutrino_mask]
    particle_ids          = particle_ids          [neutrino_mask]
    particle_pdgs         = particle_pdgs         [neutrino_mask]
    particle_energies     = particle_energies     [neutrino_mask]
    particle_parents      = particle_parents      [neutrino_mask]
    particle_interactions = particle_interactions [neutrino_mask]

    from truth_dumper import dump_truth_info
    particle_df = dump_truth_info(input_data, energy_cut, ignore_pdgs = neutrino_pdgs)

    def fill_trajectory(df, id):
        row_id = 0
        for i in range(len(particle_x)):
            if particle_ids[i] == id:
                df.loc[row_id] = [
                    particle_x[i]/10,
                    particle_y[i]/10,
                    particle_z[i]/10,
                    particle_t[i],
                ]
                row_id += 1
        # df = df.sort_values('t')
        df = df.reset_index()
        return df

    trajectories = {}
    for _, row in particle_df.iterrows():
        if row['energy'] < energy_cut:
            continue
        df = pd.DataFrame(columns=['x', 'y', 'z', 't'])
        trajectories[row['id']] = fill_trajectory(df, row['id'])
        rprint(trajectories[row['id']])
    rprint(trajectories.keys())

    view_str = f'is_{view}'
    fiber_x = ak.to_numpy(geom_data['fiber_x_min'][geom_data[view_str]])
    fiber_y = ak.to_numpy(geom_data['fiber_y_min'][geom_data[view_str]])
    fiber_z = ak.to_numpy(geom_data['fiber_z_min'][geom_data[view_str]])

    max_hit_time = np.max(hit_t)
    min_hit_time = np.min(hit_t)
    max_part_time = np.max(particle_t)
    min_part_time = np.min(particle_t)

    max_time = np.max([max_part_time, max_hit_time])

    times = np.arange(0, cut_ns, 2)
    times = np.append(times, np.arange(cut_ns, max_time+1, print_every_ns/fps))

    max_hit_x   = np.max(hit_x)
    min_hit_x   = np.min(hit_x)
    max_part_x  = np.max(particle_x)
    min_part_x  = np.min(particle_x)
    max_fiber_x = np.max(fiber_x)
    min_fiber_x = np.min(fiber_x)

    max_hit_y   = np.max(hit_y)
    min_hit_y   = np.min(hit_y)
    max_part_y  = np.max(particle_y)
    min_part_y  = np.min(particle_y)
    max_fiber_y = np.max(fiber_y)
    min_fiber_y = np.min(fiber_y)

    max_hit_z   = np.max(hit_z)
    min_hit_z   = np.min(hit_z)
    max_part_z  = np.max(particle_z)
    min_part_z  = np.min(particle_z)
    max_fiber_z = np.max(fiber_z)
    min_fiber_z = np.min(fiber_z)

    from rich.table import Table

    td = Table('Type', 'Min', 'Max', title="Dimensions (ns and mm)")
    td.add_row('hit time'     , f'{min_hit_time :0.2f}', f'{max_hit_time :0.2f}')
    td.add_row('particle time', f'{min_part_time:0.2f}', f'{max_part_time:0.2f}')
    td.add_row()
    td.add_row('hit x'        , f'{min_hit_x    :0.2f}', f'{max_hit_x    :0.2f}')
    td.add_row('particle x'   , f'{min_part_x   :0.2f}', f'{max_part_x   :0.2f}')
    td.add_row('fiber x'      , f'{min_fiber_x  :0.2f}', f'{max_fiber_x  :0.2f}')
    td.add_row()
    td.add_row('hit y'        , f'{min_hit_y    :0.2f}', f'{max_hit_y    :0.2f}')
    td.add_row('particle y'   , f'{min_part_y   :0.2f}', f'{max_part_y   :0.2f}')
    td.add_row('fiber y'      , f'{min_fiber_y  :0.2f}', f'{max_fiber_y  :0.2f}')
    td.add_row()
    td.add_row('hit z'        , f'{min_hit_z    :0.2f}', f'{max_hit_z    :0.2f}')
    td.add_row('particle z'   , f'{min_part_z   :0.2f}', f'{max_part_z   :0.2f}')
    td.add_row('fiber z'      , f'{min_fiber_z  :0.2f}', f'{max_fiber_z  :0.2f}')

    rprint(td)

    rprint(f'''Number of hits:
- x: {hit_x.shape[0]} hits
- y: {hit_x.shape[0]} hits
- z: {hit_z.shape[0]} hits
''')

    def add_padding(number, side):
        if not side in ['left', 'right']:
            raise RuntimeError(f'"side" should be "left" or "right". You provided "{side}"')
        if   number>0 and side=="right": number = number * 1.
        elif number<0 and side=="right": number = number * 1.
        elif number>0 and side=="left" : number = number * 1.
        elif number<0 and side=='left' : number = number * 1.
        return number

    max_x = add_padding(np.max([max_hit_x, max_part_x]), "right")
    max_y = add_padding(np.max([max_hit_y, max_part_y]), "right")
    max_z = add_padding(np.max([max_hit_z, max_part_z]), "right")


    min_x = add_padding(np.min([min_hit_x, min_part_x]), "left")
    min_y = add_padding(np.min([min_hit_y, min_part_y]), "left")
    min_z = add_padding(np.min([min_hit_z, min_part_z]), "left")

    mask_x_1 = fiber_x<max_x
    mask_x_2 = fiber_x>min_x
    mask_x = mask_x_1 * mask_x_2

    mask_z_1 = fiber_z<max_z
    mask_z_2 = fiber_z>min_z
    mask_z = mask_z_1 * mask_z_2

    bins_x = np.unique(np.sort(fiber_x[mask_x]))
    bins_z = np.unique(np.sort(fiber_z[mask_z]))

    rprint(f'''Size of the 3D histogram:
- x: {bins_x.shape[0]} bins {np.min(bins_x)} -> {np.max(bins_x)}
- z: {bins_z.shape[0]} bins {np.min(bins_z)} -> {np.max(bins_z)}
- total voxels: {bins_x.shape[0]*bins_z.shape[0]}
''')

    fig, ax = plt.subplots()


    def make_histogram(hit_x, hit_z, hit_t, time_ns:float=None, discard=False):
        discard = False
        # hit time:
        # | |  | |    |             |     | | ||
        #      |<---------------------|
        #      time-livetime          time
        # mask:
        # o o  | |    |             |     o o oo
        rprint(f'Time: {time_ns} ns')

        if time_ns is None:
            mask_future = hit_t>0

        elif discard:
            if time_ns<cut_ns:
                hit_lifetime_ns = hit_lifetime_ns_1
            else:
                hit_lifetime_ns = hit_lifetime_ns_2

            mask_past = hit_t>time_ns-hit_lifetime_ns
            rprint(f"{len(hit_x)} before discarding hits in the past")
            # definitely discard the hits in the past to increase speed
            hit_x = hit_x[mask_past]
            hit_z = hit_z[mask_past]
            hit_t = hit_t[mask_past]
            rprint(f"{len(hit_x)} after discarding hits in the past")
            mask_future = hit_t<time_ns
        else:
            if time_ns<cut_ns:
                hit_lifetime_ns = hit_lifetime_ns_1
            else:
                hit_lifetime_ns = hit_lifetime_ns_2
            mask_past = hit_t>time_ns-hit_lifetime_ns
            mask_future = hit_t<time_ns
            mask_future = mask_future * mask_past



        histogram, _ = np.histogramdd(
            (hit_z[mask_future], hit_x[mask_future]),
            bins=(bins_z, bins_x),
            range=None,
            density=None,
            weights=None
        )
        return histogram.T

    def make_lines(time_ns:float=None):

        pass

    h2 = make_histogram(hit_x, hit_z, hit_t, None)
    max_hits = np.max(h2)



    timeText = ax.text(0.05, 0.05, '', ha="left", va="top", transform=ax.transAxes)

    frame_number = 0

    from rich.progress import Progress
    progress = Progress()
    task = progress.add_task("[green]Processing movie...", total=max_time)

    def roll(time_ns:float):
        progress.update(task, completed=time_ns)
        h = make_histogram(hit_x, hit_z, hit_t, time_ns, True)
        timeText.set_text("{:.2f} ns".format(time_ns))

        # for u in linesToUpdate[timeIndex]:
        #     trackLocal = trackList[u][i_timeCuts[timeIndex]]
        #     plotList[u].set_data(trackLocal['i_pos_x'], trackLocal['i_pos_y'])

        im.set_data(h)

    im = plt.imshow(h2, norm=colors.LogNorm(), origin='lower', extent=(min_z/10., max_z/10., min_x/10., max_x/10.))
    # Generate line plots
    lines = []
    from particle import Particle

    for id, traj in trajectories.items():
        #if not id in [2,3]: continue
        rprint(f'Plotting track #{id} of {traj.shape[0]} points ({len(traj.z)}, {len(traj.x)})')
        row = particle_df.loc[particle_df['id'] == id].iloc[0]
        pname = '$'+Particle.from_pdgid(row['pdg']).latex_name+"$"
        e = row['energy']
        line, = ax.plot(traj.z, traj.x, label=f'{pname}: {e:.0f} MeV')
        lines.append(line)
    plt.xlabel("z [cm]")
    plt.ylabel("x [cm]")
    # plt.xlim([min_x, max_x])
    # plt.ylim([min_z, max_z])
    plt.colorbar()
    plt.legend(prop={'size': 6})
    #plt.tight_layout()

    plt.savefig(output_movie+'.png', dpi=800)

    if only_image:
        exit(0)

    progress.start()
    ani = animation.FuncAnimation(
        fig,
        roll,
        frames=times,
        interval=1,
        blit=False
    )

    plt.xlabel("z [cm]")
    plt.ylabel("x [cm]")

    plt.tight_layout()

    FFwriter = animation.FFMpegWriter(fps=fps)
    ani.save(output_movie, writer=FFwriter, dpi=800)



if __name__ == "__main__":
    main()