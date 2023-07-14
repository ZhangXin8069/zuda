pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/ztest.cc -o ztest5 && mpirun.openmpi -np 5 ztest5
popd
