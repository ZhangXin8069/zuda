pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/ztest.cc -o ztest7 && mpirun.openmpi -np 7 ztest7
popd
