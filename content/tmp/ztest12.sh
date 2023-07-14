pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/ztest.cc -o ztest12 && mpirun.openmpi -np 12 ztest12
popd
