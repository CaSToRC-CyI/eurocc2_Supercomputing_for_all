# run_jupyter_gpu.slurm
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
#SBATCH --job-name=<your_job_name>
#SBATCH --time 00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=%x-%j.out
#SBATCH --account=<your_account>    # Replace
 
# Environment Setup
module load Anaconda3/2023.03-1
source ~/.bashrc
conda activate ai_env
 
# Figure out current script path
if [ -n "${SLURM_JOB_ID:-}" ] ; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
else
    SCRIPT_PATH=$(realpath "$0")
fi
SCRIPT_DIR=$(dirname $SCRIPT_PATH)

# Choose random ports
read PORT1 <<< $(shuf -i 5000-5999 -n 1 | tr '\n' ' ')

# Launches Jupyter Notebook on the compute node without opening a browser
# Notebook is bound to the randomly selected port 
jupyter notebook --no-browser --port $PORT1 &
sleep 10
 
# Define the file path
file="${SLURM_JOB_NAME}-${SLURM_JOBID}.out"
# Extract addresses using grep and regex patterns
address1=$(grep -o 'ServerApp] http://localhost:[0-9]\+/?token=[a-zA-Z0-9]\+' "$file" | awk '{print $2}')

LOGIN_HOST="cyclone.hpcf.cyi.ac.cy"
BATCH_HOST=$(hostname)
OUTPUT_FILE="$SCRIPT_DIR/connection_instructions.txt"
 
echo "##################################################################################################" > "$OUTPUT_FILE"
echo "To connect to the notebook type the following command into your local terminal:" >> "$OUTPUT_FILE"
echo "ssh -N -J ${USER}@${LOGIN_HOST} ${USER}@${BATCH_HOST} -L ${PORT1}:localhost:${PORT1}" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "After the connection is established in your local browser, go to the following addresses:" >> "$OUTPUT_FILE"
echo "Jupyter notebooks: ${address1}" >> "$OUTPUT_FILE"
echo "##################################################################################################" >> "$OUTPUT_FILE"
 
sleep infinity