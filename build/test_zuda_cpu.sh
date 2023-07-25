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
work_name="test"
tmp_name="tmp"
work_path=${_HOME}/${work_name}
tmp_path=${_HOME}/${tmp_name}

# do
pushd $_PATH
echo "###${_NAME} is running...:$(date "+%Y-%m-%d-%H-%M-%S")###"
python ./setup_zuda_cpu.py build_ext --inplace
for i in $(find ./ -type f -name "*.so"); do
    i=$(basename ${i})
    mv i ${_HOME}/lib
    echo "mv i ${_HOME}/lib"
done
pushd ${work_path}
python ${_NAME}.py
popd
echo "###${_NAME} is done......:$(date "+%Y-%m-%d-%H-%M-%S")###"
popd

# done
