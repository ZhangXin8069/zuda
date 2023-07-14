pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zrun.cc -o zrun12 && mpirun.openmpi -np 12 zrun12
popd
