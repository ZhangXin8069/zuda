pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zexample.cc -o zexample12 && mpirun.openmpi -np 12 zexample12
popd
