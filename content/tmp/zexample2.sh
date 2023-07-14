pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zexample.cc -o zexample2 && mpirun.openmpi -np 2 zexample2
popd
