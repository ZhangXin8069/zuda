pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zwork.cc -o zwork1 && mpirun.openmpi -np 1 zwork1
popd
