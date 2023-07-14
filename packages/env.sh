install_path="$(cd ~ && pwd)/external-libraries"
file_path="$(cd ~ && pwd)/work/packages"

cd ~
add0="export PYTHONPATH=${install_path}:\$PYTHONPATH"
add1="export LD_LIBRARY_PATH=${install_path}/quda/build/lib:\$LD_LIBRARY_PATH"
echo ${add0} >> $(cd ~ && pwd)/.bashrc 
echo ${add1} >> $(cd ~ && pwd)/.bashrc 
source $(cd ~ && pwd)/.bashrc
