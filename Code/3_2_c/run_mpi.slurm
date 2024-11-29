# run_mpi.slurm
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
#SBATCH --job-name=mpi
#SBATCH --output=mpi.out
#SBATCH --error=mpi.err
#SBATCH --partition=cpu
#SBATCH --ntasks=4      # Runs on 4 Processes
#SBATCH --time=00:20:00
#SBATCH --account=<your_account>    # Replace

module load OpenMPI/4.1.6-GCC-13.2.0
srun ./hello_mpi