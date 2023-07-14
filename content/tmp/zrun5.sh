pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zrun.cc -o zrun5 && mpirun.openmpi -np 5 zrun5
popd
