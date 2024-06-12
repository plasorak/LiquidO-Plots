from truth_dumper import dump_truth_info
import click

step_types = {
    1091 : "Transportation",
    1092 : "CoupleTrans",
    2001 : "CoulombScat",
    2002 : "Ionisation",
    2003 : "Brems",
    2004 : "PairProdCharged",
    2005 : "Annih",
    2006 : "AnnihToMuMu",
    2018 : "AnnihToTauTau",
    2007 : "AnnihToHad",
    2008 : "NuclearStopping",
    2009 : "ElectronGeneral",
    2010 : "Msc",
    2011 : "Rayleigh",
    2012 : "PhotoElectric",
    2013 : "Compton",
    2014 : "Conv",
    2015 : "ConvToMuMu",
    2016 : "GammaGeneral",
    2017 : "PositronGeneral",
    2049 : "MuPairByMuon",
    2021 : "Cerenkov",
    2022 : "Scintillation",
    2023 : "SynchRad",
    2024 : "TransRad",
    2025 : "SurfaceRefl",
    3031 : "OpAbsorb",
    3032 : "OpBoundary",
    3033 : "OpRayleigh",
    3034 : "OpWLS",
    3035 : "OpMieHG",
    3036 : "OpWLS2",
    2051 : "DNAElastic",
    2052 : "DNAExcit",
    2053 : "DNAIonisation",
    2054 : "DNAVibExcit",
    2055 : "DNAAttachment",
    2056 : "DNAChargeDec",
    2057 : "DNAChargeInc",
    2058 : "DNAElecSolv",
    6059 : "DNAMolecDecay",
    1060 : "ITTransport",
    1061 : "DNABrownTrans",
    2062 : "DNADoubleIoni",
    2063 : "DNADoubleCap",
    2064 : "DNAIoniTransfer",
    9065 : "DNAStaticMol",
    9066 : "DNAScavenger",
    4111 : "HadElastic",
    4116 : "NeutronGeneral",
    4121 : "HadInelastic",
    4131 : "HadCapture",
    4132 : "MuAtomCapture",
    4141 : "HadFission",
    4151 : "HadAtRest",
    4161 : "HadCEX",
    6201 : "Decay",
    6202 : "DecayWSpin",
    6203 : "DecayPiSpin",
    6210 : "DecayRadio",
    6211 : "DecayUnKnown",
    6221 : "DecayMuAtom",
    6231 : "DecayExt",
    7401 : "StepLimiter",
    7402 : "UsrSepcCuts",
    7403 : "NeutronKiller",
    10491: "ParallelWorld",
}

def stepper(mc_truth, track_id, with_children):
    query = f'track_id == {track_id} or parent_id == {track_id}' if with_children else f'track_id == {track_id}'
    mc_truth_particle = mc_truth.query(query)
    mc_truth_particle = mc_truth_particle.sort_values(by=['i_time', 'track_id'])
    mc_truth_particle = mc_truth_particle.reset_index(drop=True)
    mc_truth_particle = mc_truth_particle.drop(columns=['event_number'])

    from rich import print as pprint
    from rich.table import Table
    from particle import Particle

    t = Table(title=f"Track #{track_id} G4 steps")
    for column in mc_truth_particle.columns:
        t.add_column(column)

    for _, row in mc_truth_particle.iterrows():
        pre = "  " if row['parent_id'] == track_id else ""
        p = Particle.from_pdgid(int(row['i_particle']))
        t.add_row(
            pre+str(int(row['track_id'])), str(int(row['parent_id'])),
            step_types.get(abs(int(row['interaction_id'])), f"{row['interaction_id']:.0f}"),
            p.name,
            f"{row['i_pos_x']:.2f}", f"{row['i_pos_y']:0.2f}", f"{row['i_pos_z']:.2f}",
            f"{row['i_dE']:0.2f}", f"{row['i_E']-p.mass:.2f}", f"{row['i_time']:0.2f}"
        )
    pprint(t)


@click.command()
@click.argument('input_data', type=click.Path(exists=True), nargs=1)
@click.argument('track_id', type=int, nargs=-1)
@click.option('--with-children', is_flag=True, default=False)
def main(input_data, track_id, with_children):
    import uproot
    import pandas as pd

    mc_truth  = uproot.open(input_data)['mc_truth'].arrays(library="pd")
    for i in track_id:
        stepper(mc_truth, i, with_children)

if __name__ == "__main__":
    main()