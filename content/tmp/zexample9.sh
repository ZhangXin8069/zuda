pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zexample.cc -o zexample9 && mpirun.openmpi -np 9 zexample9
popd
