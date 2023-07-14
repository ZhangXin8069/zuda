pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/ztest.cc -o ztest4 && mpirun.openmpi -np 4 ztest4
popd
