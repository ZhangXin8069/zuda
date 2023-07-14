pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zwork.cc -o zwork15 && mpirun.openmpi -np 15 zwork15
popd
