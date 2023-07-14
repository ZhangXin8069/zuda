pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zwork.cc -o zwork2 && mpirun.openmpi -np 2 zwork2
popd
