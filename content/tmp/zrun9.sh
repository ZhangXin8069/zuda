pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zrun.cc -o zrun9 && mpirun.openmpi -np 9 zrun9
popd
