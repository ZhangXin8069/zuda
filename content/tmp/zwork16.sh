pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zwork.cc -o zwork16 && mpirun.openmpi -np 16 zwork16
popd
