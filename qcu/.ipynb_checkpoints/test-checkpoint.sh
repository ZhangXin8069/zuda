bash ./install.sh
pushd /home/aistudio/external-libraries/PyQuda
pip install -U . -t /home/aistudio/external-libraries
popd
python ./test.dslash.qcu.py
