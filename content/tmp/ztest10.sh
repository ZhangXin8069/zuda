pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/ztest.cc -o ztest10 && mpirun.openmpi -np 10 ztest10
popd
