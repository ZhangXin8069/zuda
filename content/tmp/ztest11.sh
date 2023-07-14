pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/ztest.cc -o ztest11 && mpirun.openmpi -np 11 ztest11
popd
