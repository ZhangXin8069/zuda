pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zrun.cc -o zrun14 && mpirun.openmpi -np 14 zrun14
popd
