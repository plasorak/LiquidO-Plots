#!/bin/bash

for f in *.root;
do
    echo $f
    python movie-maker.py ${f} ${f}.movie.mp4 --energy-cut 10
done