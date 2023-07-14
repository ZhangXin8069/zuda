pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zrun.cc -o zrun11 && mpirun.openmpi -np 11 zrun11
popd
