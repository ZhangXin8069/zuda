pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zrun.cc -o zrun2 && mpirun.openmpi -np 2 zrun2
popd
