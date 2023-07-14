pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zwork.cc -o zwork10 && mpirun.openmpi -np 10 zwork10
popd
