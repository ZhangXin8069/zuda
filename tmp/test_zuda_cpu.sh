pushd /public/home/zhangxin/zuda/tmp
rm -rf ./build
python ./setup_test_zuda_cpu.py build_ext --inplace
for i in ; do
       mv 16 /public/home/zhangxin/zuda/lib
       echo mv 16 /public/home/zhangxin/zuda/lib
done
rm -rf ./build
pushd /public/home/zhangxin/zuda/test
rm -rf lib
ln -s ../lib/ .
python ./test_test_zuda_cpu.py
popd

