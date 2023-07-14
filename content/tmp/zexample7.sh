pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zexample.cc -o zexample7 && mpirun.openmpi -np 7 zexample7
popd
