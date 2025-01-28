#!/bin/bash

for L in $(seq 100 100 900) $(seq 1000 1000 9000) $(seq 10000 10000 100000); do
  for a in 0.01 $(seq 0.0 0.20 5.00); do
    echo "L=$L, a=$a"
    cat config.ini.tmp2 | sed "s/LVAL/$L/g" | sed "s/ALPHAVAL/$a/g" > config2.ini  
    chook -n 10 WP config2.ini 
    rm config2.ini
  done
done
