pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zwork.cc -o zwork4 && mpirun.openmpi -np 4 zwork4
popd
