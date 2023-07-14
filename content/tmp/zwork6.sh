pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zwork.cc -o zwork6 && mpirun.openmpi -np 6 zwork6
popd
