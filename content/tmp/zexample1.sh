pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zexample.cc -o zexample1 && mpirun.openmpi -np 1 zexample1
popd
