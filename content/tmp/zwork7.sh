pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zwork.cc -o zwork7 && mpirun.openmpi -np 7 zwork7
popd
