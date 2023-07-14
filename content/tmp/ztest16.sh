pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/ztest.cc -o ztest16 && mpirun.openmpi -np 16 ztest16
popd
