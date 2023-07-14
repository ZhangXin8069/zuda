pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zrun.cc -o zrun16 && mpirun.openmpi -np 16 zrun16
popd
