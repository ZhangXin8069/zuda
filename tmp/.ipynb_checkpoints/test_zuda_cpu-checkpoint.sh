pushd /home/aistudio/work/zuda/build
rm -rf ./build
python ./setup_test_zuda_cpu.py build_ext --inplace
for i in $(find ./ -type f -name *.so); do
       mv ${i} /home/aistudio/work/zuda/lib
       echo "mv ${i} /home/aistudio/work/zuda/lib"
done
rm -rf ./build
popd
pushd /home/aistudio/work/zuda/test
rm -rf lib
ln -s ../lib/ .
python ./test_zuda_cpu.py
popd

