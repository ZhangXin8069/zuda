pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zexample.cc -o zexample14 && mpirun.openmpi -np 14 zexample14
popd
