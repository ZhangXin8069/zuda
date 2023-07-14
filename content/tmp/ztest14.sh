pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/ztest.cc -o ztest14 && mpirun.openmpi -np 14 ztest14
popd
