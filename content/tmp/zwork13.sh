pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zwork.cc -o zwork13 && mpirun.openmpi -np 13 zwork13
popd
