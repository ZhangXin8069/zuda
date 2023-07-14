pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zrun.cc -o zrun1 && mpirun.openmpi -np 1 zrun1
popd
