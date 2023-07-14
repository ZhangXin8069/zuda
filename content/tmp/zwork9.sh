pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zwork.cc -o zwork9 && mpirun.openmpi -np 9 zwork9
popd
