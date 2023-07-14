pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zrun.cc -o zrun13 && mpirun.openmpi -np 13 zrun13
popd
