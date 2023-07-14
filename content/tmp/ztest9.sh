pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/ztest.cc -o ztest9 && mpirun.openmpi -np 9 ztest9
popd
