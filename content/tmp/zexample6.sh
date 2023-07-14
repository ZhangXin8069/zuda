pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zexample.cc -o zexample6 && mpirun.openmpi -np 6 zexample6
popd
