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


# TODO: on the dunegpvm
# - Fix fiber coordinates at the centre of the detector
# - Fix hits' is_xz is_yz
# - Generate more events
# TODO: here
# - Add truth information

@click.command()
@click.argument('input_data', type=click.Path(exists=True))
@click.argument('output_movie', type=click.Path())
def main(input_data, output_movie):

    hit_lifetime = 2.
    seconds_per_ns = 0.5
    fps = 10

    if output_movie[-4:] != ".mp4":
        raise RuntimeError('output file must be finishing with .mp4')

    hit_data  = uproot.open(input_data)['op_hits' ].arrays()
    geom_data = uproot.open(input_data)['geom'    ].arrays()
    mc_truth  = uproot.open(input_data)['mc_truth'].arrays()

    hit_x = ak.to_numpy(hit_data['h_pos_x'])
    hit_z = ak.to_numpy(hit_data['h_pos_z'])
    hit_t = ak.to_numpy(hit_data['h_time' ])

    fiber_x = ak.to_numpy(geom_data['fiber_x_min'][geom_data['is_yz']]*1.)
    fiber_z = ak.to_numpy(geom_data['fiber_z_min'][geom_data['is_yz']]*1.)

    max_time = np.max(hit_t)
    min_time = np.min(hit_t)

    times = np.arange(0, max_time+1, seconds_per_ns/fps)

    min_x = np.min(hit_x)
    max_x = np.max(hit_x)

    min_z = np.min(hit_z)
    max_z = np.max(hit_z)


    rprint(f'''
Min time: {min_time} ns
Max time: {max_time} ns
''')

    rprint(f'''Number of hits:
- x: {hit_x.shape[0]} hits
- z: {hit_z.shape[0]} hits
''')

    mask_x_1 = fiber_x<max_x
    mask_x_2 = fiber_x>min_x
    mask_x = mask_x_1 * mask_x_2

    mask_z_1 = fiber_z<max_z
    mask_z_2 = fiber_z>min_z
    mask_z = mask_z_1 * mask_z_2

    bins_x = np.unique(np.sort(fiber_x[mask_x]))
    bins_z = np.unique(np.sort(fiber_z[mask_z]))


    rprint(bins_x)
    rprint(bins_z)

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
            mask1 = hit_t>time_ns-hit_lifetime
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
            mask_past = hit_t>time_ns-hit_lifetime
            rprint(f"{len(hit_x)} before discarding hits in the past")
            # definitely discard the hits in the past to increase speed
            hit_x = hit_x[mask_past]
            hit_z = hit_z[mask_past]
            hit_t = hit_t[mask_past]
            rprint(f"{len(hit_x)} after discarding hits in the past")
            mask_future = hit_t<time_ns
        else:
            mask_past = hit_t>time_ns-hit_lifetime
            mask_future = hit_t<time_ns
            mask_future = mask_future * mask_past



        histogram, _ = np.histogramdd(
            (hit_z[mask_future], hit_x[mask_future]),
            bins=(bins_z, bins_x),
            range=None,
            density=None,
            weights=None
        )
        rprint(histogram.shape)
        return histogram


    h2 = make_histogram(hit_x, hit_z, hit_t, None)
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
    im = plt.imshow(np.rot90(h2), norm=colors.LogNorm(), origin='lower', extent=None)

    plt.savefig(output_movie+'.png')
    # exit()
    progress.start()
    ani = animation.FuncAnimation(
        fig,
        roll,
        frames=times,
        interval=1,
        blit=False
    )

    plt.xlabel("x (mm)")
    plt.ylabel("z (mm)")
    plt.xlim([min_x, max_x])
    plt.ylim([min_z, max_z])
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