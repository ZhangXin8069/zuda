pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zrun.cc -o zrun8 && mpirun.openmpi -np 8 zrun8
popd
