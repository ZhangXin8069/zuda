pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zwork.cc -o zwork3 && mpirun.openmpi -np 3 zwork3
popd
