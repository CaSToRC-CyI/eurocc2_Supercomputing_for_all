<!--
 g08_Cyclone_Launching Jupyter.md

 CaSToRC, The Cyprus Institute

 (c) 2024 The Cyprus Institute

 Contributing Authors:
 Christodoulos Stylianou (c.stylianou@cyi.ac.cy)
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     https://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# **8. Launching Jupyter Notebooks on Cyclone Compute Nodes**

This guide demonstrates how to set up and launch a Jupyter Notebook on Cyclone **compute nodes**. Running Jupyter notebooks on compute nodes ensures that you utilize Cyclone's computational resources effectively, avoiding overloading the login node.

---

## **8.1 Why Jupyter Notebooks on HPC?**
Jupyter Notebooks offer a highly interactive environment that seamlessly combines code execution, visualizations, and narrative explanations, making them ideal for tasks like data exploration, visualization, and AI model development. Their intuitive, web-based interface simplifies complex workflows, lowering the learning curve for users across various expertise levels.

Leveraging Jupyter Notebooks on HPC systems amplifies these benefits by providing access to powerful compute resources, such as CPUs and GPUs, that can handle large-scale datasets and perform demanding AI training or numerical simulations. This integration enables users to work interactively and efficiently, tackling computational challenges beyond the capabilities of local machines.

---

## **8.2 How it Works**
Running Jupyter Notebooks on an HPC system involves allocating resources using a SLURM script and establishing a secure connection to access the notebook interface in your local browser.

### **8.2.1 Workflow Steps**
1. **Write a SLURM Script:** Create a SLURM job script specifying the resources required for your Jupyter session, such as CPUs, memory, or GPUs.
2. **Submit the Script:** Use the `sbatch` command to submit the script to the HPC scheduler, which will allocate the requested resources and launch the Jupyter Notebook server.
3. **Create an SSH Tunnel:** Establish a secure SSH tunnel to forward the notebook's port from the remote HPC system to your local machine, enabling browser access.
4. **Open the Notebook:** Use the forwarded port to access the Jupyter Notebook interface in your web browser, enabling an interactive and powerful environment for your tasks.

### **8.2.2 Establishing the SSH Tunnel**
SSH Tunneling is a method to securely forward ports from the HPC system to your local machine.

Assume job is running on `gpu01` compute node, then the command for SSH Tunneling would be:
```bash
ssh –N –J your_username@cyclone.hpcf.cyi.ac.cy your_username@gpu01 –L 8888:localhost:8888
```

where `8888` is the local and remote ports for the Jupyter Notebook.

## **8.3 Example SLURM Script**

Here is an example SLURM script that follows the workflow described above and launches a Jupyter Notebook on a GPU compute node:

```bash
#!/bin/bash
#SBATCH --job-name=<your_job_name>
#SBATCH --time 00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=%x-%j.out
#SBATCH --account=<your_account>
 
# Environment Setup
module load Anaconda3/2023.03-1
source ~/.bashrc
conda activate your_environment
 
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
```

---

### **8.3.1 Script Breakdown**
The script demonstrates how to launch a Jupyter Notebook on Cyclone's compute nodes using SLURM. It begins by specifying job parameters, such as node allocation, runtime, partition, and resources (CPUs, GPUs). The script sets up the environment by loading the required Anaconda module and activating a specified Conda environment. A random port is chosen dynamically to host the Jupyter Notebook, ensuring no conflicts with other jobs.

The Jupyter Notebook is launched in **no-browser mode**, and the script **extracts the connection details** from the output log. It then generates SSH tunneling instructions and saves them, along with the notebook URL, in the `connection_instructions.txt` file for easy access. Finally, the script idles indefinitely to keep the notebook running until the job is manually terminated. 

---

## **8.4 Running the Script**

1. Save the script as `launch_jupyter.slurm`.
2. Submit the job to SLURM:
   ```bash
   sbatch launch_jupyter.slurm
   ```
3. After the job starts, a file named `connection_instructions.txt` will be created in the same directory. This file contains the SSH tunneling command and the notebook URL.

---

## **8.5 Connecting to the Jupyter Notebook**

### **8.5.1 SSH Tunneling**:
1. Locate the tunneling command in `connection_instructions.txt`. It looks like:
   ```bash
   ssh -N -J <username>@cyclone.hpcf.cyi.ac.cy <username>@<batch-node> -L <port>:localhost:<port>
   ```
2. Open a **new** terminal and run this command on your **local** machine.

### **8.5.2 Accessing the Notebook**:
1. Open a browser on your local machine.
2. Navigate to the URL provided in `connection_instructions.txt`. It typically looks like:
   ```
   http://localhost:<port>/?token=<token>
   ```

---

## **8.6 Best Practices**

1. **Resource Allocation**: Request only the necessary resources (CPUs, GPUs, memory) to optimize cluster usage.

2. **Limit Notebook Runtime**: Use the `--time` flag to set a reasonable runtime for your Jupyter session.

3. **Clean Up**: Ensure the job completes cleanly and terminates the Jupyter process on the compute node.

4. **Respect Others**: **Avoid running Jupyter on the login node**; always use compute nodes via SLURM to prevent resource contention.

---