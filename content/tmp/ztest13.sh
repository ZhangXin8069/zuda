pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/ztest.cc -o ztest13 && mpirun.openmpi -np 13 ztest13
popd
