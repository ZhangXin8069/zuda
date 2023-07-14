pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zwork.cc -o zwork5 && mpirun.openmpi -np 5 zwork5
popd
