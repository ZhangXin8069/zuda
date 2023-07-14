pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zexample.cc -o zexample5 && mpirun.openmpi -np 5 zexample5
popd
