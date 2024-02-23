#!/bin/bash

for f in /dune/app/users/plasorak/LiquidO/data/geant4/wider/event_*_data.root;
do
    echo
    echo
    echo
    echo
    echo
    echo
    echo $f
    echo
    echo
    echo
    echo
    echo
    echo
    python movie-maker.py ${f} ${f}.movie.mp4 --energy-cut 10
done
