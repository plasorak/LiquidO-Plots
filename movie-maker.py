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
def main(input_data, output_movie, only_image, view):

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

    if view == 'xz':
        hit_x = hit_x[hit_is_xz]
        hit_y = hit_y[hit_is_xz]
        hit_z = hit_z[hit_is_xz]
        hit_t = hit_t[hit_is_xz]
    elif view == 'yz':
        hit_x = hit_x[hit_is_yz]
        hit_y = hit_y[hit_is_yz]
        hit_z = hit_z[hit_is_yz]
        hit_t = hit_t[hit_is_yz]

    rprint(f'nhits {len(hit_x)}')

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
    neutrino_mask =  particle_pdgs !=  12
    neutrino_mask *= particle_pdgs != -12
    neutrino_mask *= particle_pdgs !=  14
    neutrino_mask *= particle_pdgs != -14
    neutrino_mask *= particle_pdgs !=  16
    neutrino_mask *= particle_pdgs != -16
    neutrino_mask *= particle_pdgs != 2112 # rm neutron too
    #neutrino_mask *= particle_pdgs != 0 # rm weird crap

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
    dump_truth_info(input_data)

    view_str = f'is_{view}'
    fiber_x = ak.to_numpy(geom_data['fiber_x_min'][geom_data[view_str]])
    fiber_y = ak.to_numpy(geom_data['fiber_y_min'][geom_data[view_str]])
    fiber_z = ak.to_numpy(geom_data['fiber_z_min'][geom_data[view_str]])

    max_hit_time = np.max(hit_t)
    min_hit_time = np.min(hit_t)
    max_part_time = np.max(particle_t)
    min_part_time = np.min(particle_t)

    max_time = np.max([max_part_time, max_hit_time])

    times = np.arange(0, 200, 2)
    times = np.append(times, np.arange(200, max_time+1, print_every_ns/fps)
    rprint(times)
    #    exit()

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

    max_x = np.max([max_hit_x, max_part_x])
    max_y = np.max([max_hit_y, max_part_y])
    max_z = np.max([max_hit_z, max_part_z])

    min_x = np.min([min_hit_x, min_part_x])
    min_y = np.min([min_hit_y, min_part_y])
    min_z = np.min([min_hit_z, min_part_z])

    mask_x_1 = fiber_x<max_x
    mask_x_2 = fiber_x>min_x
    mask_x = mask_x_1 * mask_x_2

    mask_z_1 = fiber_z<max_z
    mask_z_2 = fiber_z>min_z
    mask_z = mask_z_1 * mask_z_2

    bins_x = np.unique(np.sort(fiber_x[mask_x]))
    bins_z = np.unique(np.sort(fiber_z[mask_z]))

    # rprint(bins_x)
    # rprint(bins_z)

    rprint(f'''Size of the 3D histogram:
- x: {bins_x.shape[0]} bins {np.min(bins_x)} -> {np.max(bins_x)}
- z: {bins_z.shape[0]} bins {np.min(bins_z)} -> {np.max(bins_z)}
- total voxels: {bins_x.shape[0]*bins_z.shape[0]}
''')
    # mc_truth = ak.to_dataframe(mc_truth)
    # from icecream import ic
    # ic(type(mc_truth))
    # track_ids = mc_truth['track_id'].unique()
    # all_particle_data_time_ordered = []

    # for i, tid in enumerate(track_ids):
    #     particle_data_time_ordered = mc_truth[mc_truth['track_id'] == tid]
    #     particle_data_time_ordered = particle_data_time_ordered.sort_values(by=['i_time'])
    #     first_step = particle_data_time_ordered.loc[0,:]
    #     rprint(f"Considering particle #{i} of {len(track_ids)}: {Particle.from_pdgid(first_step.i_particle).name}, of energy: {first_step.i_E} GeV")
    #     all_particle_data_time_ordered.append(particle_data_time_ordered)

    #     trackUpdate = np.searchsorted(times, track['i_time'], side="right")
    #     updateIndices = np.zeros_like(times)
    #     updateIndices[np.array(list(set(trackUpdate))) - 1] = 1
    #     updateLineInfo.append(updateIndices)
    # exit()
    # updateLineInfo = np.array(updateLineInfo)
    # updateLineInfo = updateLineInfo.T
    # linesToUpdate = [np.where(updateLineInfo[timeIndex])[0] for timeIndex in range(0, len(times) - 1)]

    # plotList = []
    # labelledCheck = []

    # for l in range(0, len(trackList)):
    #     lbl = trackList[l]['i_particle'].to_list()[0]
    #     displayLbl = latexDict[trackList[l]['i_particle'].to_list()[0]]
    #     trackList[l]['i_particle'].replace(pdgToNameDict, inplace=True)
    #     col = colorDict[lbl]
    #     lw = 1.5
    #     if lbl == 22:
    #         lw = 0.5
    #     if lbl not in labelledCheck:
    #         tempPlot, = plt.plot([], [], label=displayLbl, color=col, lw=lw)
    #     else:
    #         tempPlot, = plt.plot([], [], color=col, lw=lw)
    #     labelledCheck.append(lbl)
    #     plotList.append(tempPlot)




    fig, ax = plt.subplots()


    def make_histogram(time_ns:float=None):
        # hit time:
        # | |  | |    |             |     | | ||
        #      |<---------------------|
        #      time-livetime          time
        # mask:
        # o o  | |    |             |     o o oo

        if time_ns is None:
            mask = hit_t>0
        else:
            mask1 = hit_t>time_ns-hit_lifetime_ns
            mask2 = hit_t<time_ns
            mask = mask1 * mask2

        histogram, _ = np.histogramdd((hit_x[mask], hit_z[mask]), bins=(bins_x, bins_z), range=None, density=None, weights=None)
        return histogram

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
            mask_past = hit_t>time_ns-hit_lifetime_ns
            rprint(f"{len(hit_x)} before discarding hits in the past")
            # definitely discard the hits in the past to increase speed
            hit_x = hit_x[mask_past]
            hit_z = hit_z[mask_past]
            hit_t = hit_t[mask_past]
            rprint(f"{len(hit_x)} after discarding hits in the past")
            mask_future = hit_t<time_ns
        else:
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
        return histogram


    h2 = make_histogram(hit_x, hit_z, hit_t, None)
    # rprint(h2)
    max_hits = np.max(h2)

    timeText = ax.text(0.05, 0.05, str(0), ha="left", va="top", transform=ax.transAxes)

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


    # m = np.float64(0.)
    # for t in times:
    #     m = np.max([np.max(make_histogram(hit_x, hit_z, hit_t, time_ns, t)), m])

    # im = plt.imshow(h2, vmin=0.1, vmax=int(m), colors=colors.LogNorm())
    im = plt.imshow(h2, norm=colors.LogNorm(), origin='lower', extent=(min_z/10., max_z/10., min_x/10., max_x/10.))
    plt.savefig(output_movie+'.png')

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
    # plt.xlim([min_x, max_x])
    # plt.ylim([min_z, max_z])
    plt.colorbar()
    plt.tight_layout()

    FFwriter = animation.FFMpegWriter(fps=fps)
    ani.save(output_movie, writer=FFwriter, dpi=400)




    # x_coord = []
    # y_coord = []
    # z_coord = []
    # values = []

    # #exit(0)
    # fig = plt.figure()
    # ax = fig.add_subplot()

    # colorsMap='jet'
    # cm = plt.get_cmap(colorsMap)

    # cNorm = matplotlib.colors.Normalize(vmin=min(values), vmax=max(values))
    # scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    # ax.scatter(z_coord, x_coord, marker='o', c=scalarMap.to_rgba(values))
    # scalarMap.set_array(values)

    # ax.set_xlabel('Z')
    # # ax.set_ylabel('Y Label')
    # ax.set_ylabel('X')

    # plt.show()
    # exit(0)

    # #exit(0)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    # colorsMap='jet'
    # cm = plt.get_cmap(colorsMap)

    # cNorm = matplotlib.colors.Normalize(vmin=min(values), vmax=max(values))
    # scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    # ax.scatter(x_coord, y_coord, z_coord, marker='o', c=scalarMap.to_rgba(values))
    # scalarMap.set_array(values)


    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')

    # plt.show()

if __name__ == "__main__":
    main()