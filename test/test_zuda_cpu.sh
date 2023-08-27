# source
_PATH=$(
    cd "$(dirname "$0")"
    pwd
)
echo "PATH:"$_PATH
pushd ${_PATH}/../
source ./env.sh
popd

# init
_NAME=$(basename "$0")
name="zuda_cpu"
work_name="test"
tmp_name="build"
work_path=${_HOME}/${work_name}
tmp_path=${_HOME}/${tmp_name}

# do
pushd ${tmp_path}
echo "###${_NAME} is running...:$(date "+%Y-%m-%d-%H-%M-%S")###"
rm -rf ./build
python ./setup_${name}.py build_ext --inplace
for i in $(find ./ -type f -name "*.so"); do
    mv ${i} ${_HOME}/lib
    echo "mv ${i} ${_HOME}/lib"
done
rm -rf ./build
pushd ${work_path}
rm -rf lib
ln -s ../lib/ .
python ./test_${name}.py
popd
echo "###${_NAME} is done......:$(date "+%Y-%m-%d-%H-%M-%S")###"
popd

# done
