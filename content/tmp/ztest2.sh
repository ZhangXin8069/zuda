pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/ztest.cc -o ztest2 && mpirun.openmpi -np 2 ztest2
popd
