#!/bin/bash


source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
setup root v6_26_06b -q e20:p3913:prof
setup geant4 v4_11_1_p01ba -q e26:prof
setup cmake v3_27_4
setup gdb v13_1
setup genie v3_04_00f  -q e26:prof
setup genie_xsec v3_04_00 -q G1810a0211a:e1000:k250
setup valgrind v3_21_0