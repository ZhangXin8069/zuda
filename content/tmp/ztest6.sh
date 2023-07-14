pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/ztest.cc -o ztest6 && mpirun.openmpi -np 6 ztest6
popd
