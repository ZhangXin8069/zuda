bash ./install.sh
pushd ${HOME}/external-libraries/PyQuda-master
pip install -U . -t ${HOME}/external-libraries
popd
python ./test.dslash.qcu.py
