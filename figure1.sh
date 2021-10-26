#!/bin/bash

for d in 20 30;do
  for N in 10 20;do
    python3 experiment.py -d $d -N $N
    python3 plot.py -d $d -N $N
  done
done
