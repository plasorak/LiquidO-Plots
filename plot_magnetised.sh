#!/bin/bash

# cp ../G4/build/magnetised_0.4T_r0_0830_data.root  magnetised_0.4T_r0_0830_data.root
# python /afs/cern.ch/work/p/plasorak/LiquidO/Plots/golden_plot.py --plot-options plot_options.json /eos/user/p/plasorak/LiquidO/g4_0906/r0/event_0830.mac_data.root not_magnetised_r0_0830.png
python /afs/cern.ch/work/p/plasorak/LiquidO/Plots/golden_plot.py --plot-options plot_options.json magnetised_0.4T_r0_0830_data.root magnetised_0.4T_r0_0830.png
# python /afs/cern.ch/work/p/plasorak/LiquidO/Plots/golden_plot.py --plot-options plot_options.json /eos/user/p/plasorak/LiquidO/g4_0906/r1/event_0129.mac_data.root not_magnetised_r1_0129.png
