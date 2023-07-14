pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zrun.cc -o zrun7 && mpirun.openmpi -np 7 zrun7
popd
