pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zwork.cc -o zwork14 && mpirun.openmpi -np 14 zwork14
popd
