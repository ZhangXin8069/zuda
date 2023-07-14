pushd $(cd ~ && pwd)/work/zuda
bash zuda.install.sh
python test.zuda.py
popd
