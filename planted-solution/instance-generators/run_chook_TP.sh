#!/bin/bash

for L in $(seq 100 50 500); do
  for p in $(seq 0.0 0.10 1.00); do
    echo "L=$L, p2=$p"
    cat config.ini.tmp | sed "s/LVAL/$L/g" | sed "s/P2VAL/$p/g" > config.ini  
    chook -n 10 TP config.ini 
    rm config.ini
  done
done
