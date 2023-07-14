pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/ztest.cc -o ztest1 && mpirun.openmpi -np 1 ztest1
popd
