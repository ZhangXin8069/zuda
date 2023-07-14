pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zexample.cc -o zexample15 && mpirun.openmpi -np 15 zexample15
popd
