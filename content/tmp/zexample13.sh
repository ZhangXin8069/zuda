pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zexample.cc -o zexample13 && mpirun.openmpi -np 13 zexample13
popd
