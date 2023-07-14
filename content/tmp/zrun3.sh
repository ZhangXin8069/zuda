pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zrun.cc -o zrun3 && mpirun.openmpi -np 3 zrun3
popd
