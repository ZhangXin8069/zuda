pushd /home/aistudio/work/content/tmp
mpic++.openmpi /home/aistudio/work/content/bin/zwork.cc -o zwork8 && mpirun.openmpi -np 8 zwork8
popd
