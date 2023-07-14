pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zrun.cc -o zrun15 && mpirun.openmpi -np 15 zrun15
popd
