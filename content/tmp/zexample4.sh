pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zexample.cc -o zexample4 && mpirun.openmpi -np 4 zexample4
popd
