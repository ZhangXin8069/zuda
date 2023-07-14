pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zexample.cc -o zexample10 && mpirun.openmpi -np 10 zexample10
popd
