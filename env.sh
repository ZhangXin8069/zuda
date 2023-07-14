pushd $(cd ~ && pwd)/work/packages
source env.sh 
popd
pushd $(cd ~ && pwd)/work/content
source env.sh 
popd
popd
pushd $(cd ~ && pwd)/work/configure
source env.sh 
popd
#sleep 5s
clear
