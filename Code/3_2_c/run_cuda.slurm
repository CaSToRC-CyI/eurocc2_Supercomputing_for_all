# run_cuda.slurm
#
# CaSToRC, The Cyprus Institute
#
# (c) 2024 The Cyprus Institute
#
# Contributing Authors:
# Christodoulos Stylianou (c.stylianou@cyi.ac.cy)
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
#SBATCH --job-name=cuda
#SBATCH --output=cuda.out
#SBATCH --error=cuda.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1        # Runs on 1 GPU
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --account=<your_account>    # Replace

module load CUDA/12.1.1
srun ./hello_cuda