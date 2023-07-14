pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zwork.cc -o zwork12 && mpirun.openmpi -np 12 zwork12
popd
