import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot
from rich import print as rprint
import click

@click.command()
@click.argument('input_data_numu', type=click.Path(exists=True))
@click.argument('input_data_anumu', type=click.Path(exists=True))
def main(input_data_numu, input_data_anumu):

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    interaction_pdgs_numu  = uproot.open(input_data_numu )['part'].arrays()['pdg']
    interaction_pdgs_anumu = uproot.open(input_data_anumu)['part'].arrays()['pdg']

    pdgs = np.unique(np.concatenate((interaction_pdgs_numu, interaction_pdgs_anumu)))
    pdgs_numu = []
    pdgs_anumu = []

    for pdg in pdgs:
        pdgs_numu .append(np.count_nonzero(interaction_pdgs_numu ==pdg)/10000.)
        pdgs_anumu.append(np.count_nonzero(interaction_pdgs_anumu==pdg)/10000.)


    from particle import Particle
    names = [f'${Particle.from_pdgid(pdg).latex_name}$' for pdg in pdgs]

    # species = ("Adelie", "Chinstrap", "Gentoo")
    # penguin_means = {
    #     'Bill Depth': (18.35, 18.43, 14.98),
    #     'Bill Length': (38.79, 48.83, 47.50),
    #     'Flipper Length': (189.95, 195.82, 217.19),
    # }

    x = np.arange(len(names))  # the label locations
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots(figsize=(10,6),layout='constrained')

    #for attribute, measurement in penguin_means.items():
    offset = width
    rects = ax.bar(x + offset, pdgs_numu, width,label=r"1 GeV $\nu_{\mu}$ interaction on $C^{12}$")
    #ax.bar_label(rects, padding=3)
    offset = width*2
    rects = ax.bar(x + offset, pdgs_anumu, width,label=r"1 GeV anti-$\nu_{\mu}$ interaction on $C^{12}$")
    #ax.bar_label(rects, padding=3)


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('N remnants')
    ax.set_title('Nuclear remnant (CC+NC)')
    ax.set_xticks(x + width, names)
    ax.set_yscale("log")
    ax.legend()

    # plt.show()

    # plt.bar(bins, pdgs_numu,  label="1 GeV $\nu_{\mu}$ interaction on $C^{12}$")
    # plt.bar(bins, pdgs_anumu, label="1 GeV anti-$\nu_{\mu}$ interaction on $C^{12}$")

    # plt.legend()
    plt.savefig('nuclear_remnants.png', dpi=300)

if __name__ == "__main__":
    main()