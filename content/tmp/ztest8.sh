pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/ztest.cc -o ztest8 && mpirun.openmpi -np 8 ztest8
popd
