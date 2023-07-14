pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zexample.cc -o zexample16 && mpirun.openmpi -np 16 zexample16
popd
