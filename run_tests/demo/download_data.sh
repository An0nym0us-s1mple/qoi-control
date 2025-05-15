#!/bin/bash

# download Hurricane
cd ./dataset
wget https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/Hurricane-ISABEL/SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz
tar xfz SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz
rm SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz
cd ./100x500x500
mv Uf48.bin.f32 ../Hurricane
mv Vf48.bin.f32 ../Hurricane
mv Wf48.bin.f32 ../Hurricane
cd .. && rm -rf 100x500x500
# download NYX
wget https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/EXASKY/NYX/SDRBENCH-EXASKY-NYX-512x512x512.tar.gz
tar xfz SDRBENCH-EXASKY-NYX-512x512x512.tar.gz
rm SDRBENCH-EXASKY-NYX-512x512x512.tar.gz
cd SDRBENCH-EXASKY-NYX-512x512x512
mv velocity_x.f32 ../NYX
mv velocity_y.f32 ../NYX
mv velocity_z.f32 ../NYX
cd .. && rm -rf SDRBENCH-EXASKY-NYX-512x512x512

# convert single-precision to double-precision
cd ..
/usr/bin/python3 ./run_tests/demo/convert.py
