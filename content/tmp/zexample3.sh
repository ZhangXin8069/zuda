pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zexample.cc -o zexample3 && mpirun.openmpi -np 3 zexample3
popd
