pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zexample.cc -o zexample11 && mpirun.openmpi -np 11 zexample11
popd
