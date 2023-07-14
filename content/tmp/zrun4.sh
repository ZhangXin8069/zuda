pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zrun.cc -o zrun4 && mpirun.openmpi -np 4 zrun4
popd
