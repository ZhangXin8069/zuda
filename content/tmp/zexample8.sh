pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zexample.cc -o zexample8 && mpirun.openmpi -np 8 zexample8
popd
