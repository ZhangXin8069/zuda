pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zwork.cc -o zwork11 && mpirun.openmpi -np 11 zwork11
popd
