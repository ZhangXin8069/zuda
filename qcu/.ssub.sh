#!/bin/bash 
#SBATCH --job-name=ssub
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
# #SBATCH --cpus-per-task=1
#SBATCH -n 1
# # SBATCH --ntasks-per-node=48
# #SBATCH --nodelist=node[3,4]
# #SBATCH --exclude=node[1,5-6]
#SBATCH --time=00-00:30:00
#SBATCH --output=ssub.out
#SBATCH --error=ssub.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhangxin8069@qq.com
#SBATCH --gres=gpu:1
source /public/home/zhangxin/env.sh

test.sh
