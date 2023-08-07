bash ./install.sh
pushd /home/aistudio/external-libraries/PyQuda-master
pip install -U . -t /home/aistudio/external-libraries
popd
python /home/aistudio/external-libraries/PyQuda-master/tests/test.dslash_eo.qcu.py
