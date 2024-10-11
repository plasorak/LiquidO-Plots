


def dump_truth_info(input_data, energy_cut, ignore_pdgs=[]):
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
            'length', 'n_steps'
        ]
    )

    def track_length(x1, x2, y1, y2, z1, z2):
        return np.sqrt(
            (x1-x2)*(x1-x2) +
            (y1-y2)*(y1-y2) +
            (z1-z2)*(z1-z2)
        )
    from anytree import Node, RenderTree

    nodes = {}
    root = Node('root')

    for ip, id in enumerate(particle_ids):
        index = particle_df[particle_df['id'] == id].index

        if particle_pdgs[ip] in ignore_pdgs:
            continue
        mass = Particle.from_pdgid(particle_pdgs[ip].item()).mass
        if mass is None:
            mass = 0

        if not index.empty:
            index = index[0]
            if particle_df['first_time'][index] > particle_t[ip]:

                particle_df.loc[index, 'first_time'] = particle_t[ip]
                particle_df.loc[index, 'x_start'] = particle_x[ip]
                particle_df.loc[index, 'y_start'] = particle_y[ip]
                particle_df.loc[index, 'z_start'] = particle_z[ip]
                particle_df.loc[index, 'first_time'] = particle_t[ip]
                particle_df.loc[index, 'length'] = track_length(
                    particle_df.loc[index, 'x_start'],
                    particle_df.loc[index, 'x_end'],
                    particle_df.loc[index, 'y_start'],
                    particle_df.loc[index, 'y_end'],
                    particle_df.loc[index, 'z_start'],
                    particle_df.loc[index, 'z_end']
                )

            if particle_df.loc[index, 'energy'] < particle_energies[ip] - mass:
                particle_df.loc[index, 'energy'] = particle_energies[ip] - mass

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
            particle_df.loc[index, 'n_steps'] += 1
        else:
            nodes[id] = Node(id)
            particle_df.loc[row_id] = [
                id,
                particle_pdgs[ip],
                Particle.from_pdgid(particle_pdgs[ip].item()).name,

                particle_energies[ip] - mass,

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

                0.,
                1,

            ]
            row_id += 1


    from rich.table import Table

    t = Table(title="Particles")
    for column in particle_df.columns:
        t.add_column(column)

    particle_df = particle_df.sort_values('id')

    particle_df = particle_df.reset_index()  # make sure indexes pair with number of rows

    n_rows = 0

    def select(row):

        # if abs(row['pdg']) == 11 and row['energy'] < 10:
        #     return False

        # if row['pdg'] == 22 and row['energy'] < 4:
        #     return False

        # if row['length'] < 1 and row['pdg'] > 3000:
        #     return False

        return True

    for _, row in particle_df.iterrows():
        if not select(row):
            continue
        n_rows += 1
        nodes[row['id']].parent = nodes[row['parent']] if row['parent'] in nodes else root


    for pre, fill, node in RenderTree(root):
        if node.name == 'root':
            continue

        row = particle_df.loc[particle_df['id'] == int(node.name)].iloc[0]

        t.add_row(
            pre + str(row['id']), str(row['pdg']),
            str(row['name']), f"{row['energy']:.3f}",
            f"{row['first_time']:.2f}", f"{row['last_time']:0.2f}",

            f"{row['x_start']:.2f}", f"{row['x_end']:0.2f}",
            f"{row['y_start']:.2f}", f"{row['y_end']:0.2f}",
            f"{row['z_start']:.2f}", f"{row['z_end']:0.2f}",

            str(row['primary']), str(row['parent']), str(row['interaction']),

            f"{row['length']:.2f}", str(row['n_steps']),

            style='red' if row['primary'] else None,


        )

    rprint(t)
    rprint(f'There are {n_rows} rows')
    return particle_df
