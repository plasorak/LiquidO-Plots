


def dump_truth_info(input_data, energy_cut, ignore_pdgs):
    import awkward as ak
    import numpy as np
    import uproot
    from rich import print as rprint
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    from particle import Particle
    import pandas as pd

    mc_truth  = uproot.open(input_data)['mc_truth'].arrays()

    particle_x            = ak.to_numpy(mc_truth['i_pos_x'])
    particle_y            = ak.to_numpy(mc_truth['i_pos_y'])
    particle_z            = ak.to_numpy(mc_truth['i_pos_z'])
    particle_t            = ak.to_numpy(mc_truth['i_time' ])
    particle_ids          = ak.to_numpy(mc_truth['track_id'])
    particle_pdgs         = ak.to_numpy(mc_truth['i_particle'])
    particle_energies     = ak.to_numpy(mc_truth['i_E'])
    particle_parents      = ak.to_numpy(mc_truth['parent_id'])
    particle_interactions = ak.to_numpy(mc_truth['interaction_id'])

    row_id = 0
    particle_df = pd.DataFrame(columns=[
            'id', 'pdg', 'name',
            'energy', 'first_time', 'last_time',
            'x_start', 'x_end',
            'y_start', 'y_end',
            'z_start', 'z_end',
            'primary', 'parent', 'interaction',
            'length',
        ]
    )

    def track_length(x1, x2, y1, y2, z1, z2):
        return np.sqrt(
            (x1-x2)*(x1-x2) +
            (y1-y2)*(y1-y2) +
            (z1-z2)*(z1-z2)
        )

    for ip, id in enumerate(particle_ids):
        index = particle_df[particle_df['id'] == id].index

        if particle_pdgs[ip] in ignore_pdgs:
            continue

        if not index.empty:
            index = index[0]
            if particle_df['first_time'][index] > particle_t[ip]:

                particle_df.loc[index, 'first_time'] = particle_t[ip]
                particle_df.loc[index, 'x_start'] = particle_x[ip]
                particle_df.loc[index, 'y_start'] = particle_y[ip]
                particle_df.loc[index, 'z_start'] = particle_z[ip]
                particle_df.loc[index, 'first_time'] = particle_t[ip]
                particle_df.loc[index, 'energy'] = particle_energies[ip]
                particle_df.loc[index, 'length'] = track_length(
                    particle_df.loc[index, 'x_start'],
                    particle_df.loc[index, 'x_end'],
                    particle_df.loc[index, 'y_start'],
                    particle_df.loc[index, 'y_end'],
                    particle_df.loc[index, 'z_start'],
                    particle_df.loc[index, 'z_end']
                )
            if particle_df['last_time'][index] < particle_t[ip]:
                particle_df.loc[index, 'last_time'] = particle_t[ip]
                particle_df.loc[index, 'x_end'] = particle_x[ip]
                particle_df.loc[index, 'y_end'] = particle_y[ip]
                particle_df.loc[index, 'z_end'] = particle_z[ip]
                particle_df.loc[index, 'length'] = track_length(
                    particle_df.loc[index, 'x_start'],
                    particle_df.loc[index, 'x_end'],
                    particle_df.loc[index, 'y_start'],
                    particle_df.loc[index, 'y_end'],
                    particle_df.loc[index, 'z_start'],
                    particle_df.loc[index, 'z_end']
                )
        else:
            particle_df.loc[row_id] = [
                id,
                particle_pdgs[ip],
                Particle.from_pdgid(particle_pdgs[ip].item()).name,

                particle_energies[ip],

                particle_t[ip],
                particle_t[ip],

                particle_x[ip],
                particle_x[ip],

                particle_y[ip],
                particle_y[ip],

                particle_z[ip],
                particle_z[ip],

                True if particle_parents[ip] == 0 else False,
                particle_parents[ip],
                particle_interactions[ip],

                0.
            ]
            row_id += 1


    from rich.table import Table

    t = Table(title="Particles")
    for column in particle_df.columns:
        t.add_column(column)

    particle_df = particle_df.sort_values('id')

    particle_df = particle_df.reset_index()  # make sure indexes pair with number of rows

    for _, row in particle_df.iterrows():
        Particle.from_pdgid(particle_pdgs[ip].item()).mass
        if row['energy'] < energy_cut or abs(particle_pdgs[ip].item()) >3000:
            continue
        t.add_row(
            str(row['id']), str(row['pdg']),
            row['name'], f"{row['energy']:.1f}",
            f"{row['first_time']:.2f}", f"{row['last_time']:0.2f}",

            f"{row['x_start']:.2f}", f"{row['x_end']:0.2f}",
            f"{row['y_start']:.2f}", f"{row['y_end']:0.2f}",
            f"{row['z_start']:.2f}", f"{row['z_end']:0.2f}",

            str(row['primary']), str(row['parent']), str(row['interaction']),

            f"{row['length']:.2f}",

            style='red' if row['primary'] else None,


        )

    rprint(t)
    return particle_df