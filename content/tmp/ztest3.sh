pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/ztest.cc -o ztest3 && mpirun.openmpi -np 3 ztest3
popd
