pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/ztest.cc -o ztest15 && mpirun.openmpi -np 15 ztest15
popd
