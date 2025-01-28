#!/bin/bash

for L in 10; do
  for k in 3 4 5; do
    echo "L=$L, k=$k"
    cat config.ini.tmp3 | sed "s/NVAL/$L/g" | sed "s/KVAL/$k/g" > config3.ini  
    chook -n 10 XORSAT config3.ini 
    rm config3.ini
  done
done
