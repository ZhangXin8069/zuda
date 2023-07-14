pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zrun.cc -o zrun10 && mpirun.openmpi -np 10 zrun10
popd
