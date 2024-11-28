<!--
 g06_Cyclone_Job Submission.md

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

# 6. Submitting and Monitoring Jobs Using SLURM on Cyclone

## **6.1 Introduction to SLURM**
SLURM (Simple Linux Utility for Resource Management) is a powerful and widely-used workload manager and job scheduler for high-performance computing (HPC) systems. It allows users to request computational resources (like CPUs, memory, or GPUs) and ensures efficient allocation of these resources among users. SLURM is crucial for HPC environments like Cyclone because it maximizes resource utilization and ensures fair access to computational power for all users.

---

## **6.2 Why Use SLURM?**
1. **Resource Allocation**: Ensures fair distribution of CPU, GPU, memory, and storage resources.
2. **Job Scheduling**: Manages the execution order of jobs based on priorities, dependencies, and resource availability.
3. **Scalability**: Handles jobs from single-core computations to complex, multi-node workflows.
4. **Monitoring**: Tracks job status, execution logs, and performance metrics.
5. **Customization**: Allows users to specify resource needs and execution conditions via scripts.

---

## **6.3 SLURM Cheat Sheet: Job Management and Monitoring**
This cheat sheet covers essential SLURM commands for job management, resource allocation, and monitoring. For optimal resource utilization and respectful scheduling, remember to follow the **best practices** outlined in the guide.
| **Category**           | **Command**                      | **Description**                                                           |
| ---------------------- | -------------------------------- | ------------------------------------------------------------------------- |
| **Job Submission**     | `sbatch <script>`                | Submit a job using a submission script.                                   |
| **Cancel Job**         | `scancel <job_id>`               | Cancel a specific job using its job ID.                                   |
| **Hold Job**           | `scontrol hold <job_id>`         | Place a job on hold, preventing it from starting.                         |
| **Release Job**        | `scontrol release <job_id>`      | Release a held job, allowing it to start when resources become available. |
| **Queue Overview**     | `squeue`                         | View the status of all jobs in the queue.                                 |
| **My Queued Jobs**     | `squeue -u <your_username>`      | View jobs specific to your user account.                                  |
| **Detailed Job Info**  | `scontrol show job <job_id>`     | Show detailed information about a specific job.                           |
| **Job History**        | `sacct -j <job_id>`              | View historical job statistics and performance for a specific job.        |
| **Partition Info**     | `sinfo`                          | View available partitions, their nodes, and current states.               |
| **Node Details**       | `sinfo -N`                       | Display detailed node information for all partitions.                     |
| **Resources Per Node** | `scontrol show node <node_name>` | Display detailed resource availability for a specific node.               |

---

### **6.3.1 Job Status Symbols**
SLURM uses the following symbols to indicate the current state of a job:

| **Symbol** | **Status**   | **Description**                                                   |
| ---------- | ------------ | ----------------------------------------------------------------- |
| `PD`       | Pending      | Job is waiting for resources or dependencies to become available. |
| `R`        | Running      | Job is currently executing.                                       |
| `CG`       | Completing   | Job is completing, cleaning up resources.                         |
| `CD`       | Completed    | Job has finished successfully.                                    |
| `F`        | Failed       | Job failed to execute successfully.                               |
| `TO`       | Timeout      | Job exceeded the allocated time limit.                            |
| `CA`       | Canceled     | Job was canceled by the user or administrator.                    |
| `NF`       | Node Failure | Job failed due to a node failure.                                 |
| `ST`       | Stopped      | Job has been stopped.                                             |

---

## **6.4 Submitting Jobs to Cyclone**

### **6.4.1 Basic Job Submission**
Jobs are submitted to SLURM using a submission script, which is a Bash script with SLURM-specific directives (prefixed with `#SBATCH`). A basic example:

```bash
#!/bin/bash
#SBATCH --job-name=test_job       # Name of the job
#SBATCH --output=output.log       # Standard output log file
#SBATCH --error=error.log         # Standard error log file
#SBATCH --time=01:00:00           # Time limit (hh:mm:ss)
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=1                # Number of tasks (processes)
#SBATCH --cpus-per-task=4         # CPU cores per task
#SBATCH --mem=8G                  # Memory per node
#SBATCH --partition=cpu           # Partition (cpu or gpu)
#SBATCH --account=<your-account>  # Project account to be charged

# Your commands go below
echo "Running on node: $HOSTNAME"
```
Submit the script using:
```bash
sbatch job_script.sh
```

---

### **6.4.2 Understanding the Submission Script**
Each component of the SLURM script plays a crucial role:

1. **Job Name** (`#SBATCH --job-name`):
   - A meaningful name makes it easier to identify your job in the queue.

2. **Output and Error Logs** (`#SBATCH --output`, `#SBATCH --error`):
   - Specify files to store the standard output and error streams.

3. **Time Limit** (`#SBATCH --time`):
   - Define the maximum runtime to prevent jobs from overusing resources. Unused time is not billed.

4. **Node and Task Allocation**:
   - `#SBATCH --nodes`: Number of nodes.
   - `#SBATCH --ntasks`: Total number of tasks/processes.
   - `#SBATCH --cpus-per-task`: Number of threads per task.

5. **Memory Allocation** (`#SBATCH --mem`): Memory required per node in MB.

6. **Partition** (`#SBATCH --partition`): Choose between `cpu` (default) and `gpu`.

7. **Account** (`#SBATCH --account`): The project account to charge. 
   
   **Important Note**: Even though the resources on Cyclone are free of charge, they are still tracked and charged on a project assigned to the user's account. It is important to be cautions and use allocated resources wisely. To view your current quota across all of your active projects you can type `qhist` on the terminal. For details see the output of `qhist --help`.

8. **Commands**: Specify the application or script you want to run.

---

### **6.4.3 Best Practices for Submission Scripts**
1. **Request Only What You Need**:
   - Avoid over-requesting resources (e.g., more nodes, memory, or time than necessary).
   - Example: If your job requires only 4 CPUs, don’t request an entire 40-core node.

2. **Use `--exclusive` For Benchmarks**:
   - The `--exclusive` directive ensures that no other jobs are allocated to the same node, thereby dedicating the entire node to your job. This is useful when you want to get an accurate runtime of your application without other users affecting your run.
   - For example, the memory bandwidth is shared on each node/CPU socket, therefore if two users are on the same CPU they will be competing for memory resources.
   - To obtain exclusive access, add to the SLURM script the following;
      ```bash
      #SBATCH --exclusive
      ```

3. **Test Small Jobs First**:
   - Run short test jobs to ensure your script works before scaling to larger resources.

4. **Set Realistic Time Limits**:
   - Estimate the runtime and add a buffer. This ensures your job completes while minimizing wait time for others.
   - **Important Note**: The maximum continuous runtime allowed on Cyclone is 24 hours. If you want to run for more than 24 hours, your application must be capable of start/stop (checkpointing).

5. **Respect Others**:
   - Avoid monopolizing resources. Cyclone is shared by many users, so be considerate when submitting large jobs.

---

## **6.5 Monitoring Jobs**

### **6.5.1 Checking Job Status**
Use `squeue` to view active jobs:
```bash
squeue -u <your_username>
```
Output example:
```
JOBID   PARTITION   NAME      USER   ST   TIME    NODES   NODELIST(REASON)
12345   cpu         test_job  user1  R    00:05:12   1      cn01
```

### **6.5.2 Canceling Jobs**
Cancel a running or queued job:
```bash
scancel <job_id>
```

### **6.5.3 Viewing Completed Jobs**
Use `sacct` to see statistics of completed jobs:
```bash
sacct -j <job_id>
```

---

## **6.6 Advanced SLURM Features**

<!-- ### **6.6.1 Job Dependencies**
You can chain jobs using dependencies, ensuring one job starts only after another completes. Example:
   ```bash
   #SBATCH --dependency=afterok:<job_id>
   ```
Submit a dependent job:
   ```bash
   sbatch --dependency=afterok:12345 dependent_job.sh
   ```

### **6.6.2 Job Arrays** -->
### **6.6.1 Job Arrays**
Job arrays allow you to submit multiple similar jobs efficiently:
```bash
#!/bin/bash
#SBATCH --array=1-10                # Create a job array with indices 1 to 10
#SBATCH --job-name=array_job        # Job name
#SBATCH --output=job_%A_%a.log      # Output file (use %A for array ID, %a for task ID)
#SBATCH --time=00:30:00             # Time per job
#SBATCH --cpus-per-task=2           # CPUs per task
#SBATCH --partition=cpu             # Partition

# Commands using SLURM_ARRAY_TASK_ID
echo "Processing task ID: $SLURM_ARRAY_TASK_ID"
```
Submit the array with:
```bash
sbatch array_job.sh
```

## **6.7 Multithreading and SLURM: Handling and Best Practices**

Multithreading allows a single application to execute multiple threads in parallel, potentially improving performance by utilizing multiple CPUs or logical processors. Handling multithreading efficiently on Cyclone involves understanding how your application interacts with resources like CPUs, cores, and threads. By using SLURM directives such as `--cpus-per-task`, `--threads-per-core`, and `--hint`, you can fine-tune resource allocation for optimal performance while ensuring fair use of shared resources. This section explains how to effectively handle multithreading in your SLURM jobs.

---

### **6.7.1 Understanding Multithreading on Cyclone**
Cyclone nodes are equipped with advanced multithreading capabilities that can enhance computational performance when utilized appropriately. Each node features 80 **logical** CPUs when hyper-threading is enabled. In this mode, each of the 40 **physical** cores is capable of running two threads simultaneously, effectively doubling the number of logical CPUs. However, when hyper-threading is disabled, each physical core runs a single thread, resulting in 40 physical cores available for computation.

Multithreading is particularly beneficial for applications designed to leverage multiple threads effectively, such as those using parallel programming models like OpenMP. By enabling hyper-threading, these applications can exploit the additional logical CPUs to potentially boost performance. However, for *CPU-bound* or *memory-intensive* tasks, excessive multithreading can lead to **resource contention**, reducing efficiency. It is essential to evaluate your application's threading needs and configure SLURM job submissions accordingly to achieve optimal performance.

---

### **6.7.2 Key SLURM Options for Multithreading**
| Option                     | Description                                                                                    |
| -------------------------- | ---------------------------------------------------------------------------------------------- |
| `--cpus-per-task=<num>`    | Specifies the number of CPUs (threads) allocated per task.                                     |
| `--hint=multithread`       | Allows the use of hyper-threading (default behavior when hyper-threading is enabled).          |
| `--hint=nomultithread`     | Disables hyper-threading, limiting execution to physical cores.                                |
| `--ntasks=<num>`           | Sets the total number of tasks (often corresponds to MPI processes).                           |
| `--threads-per-core=<num>` | Explicitly sets the number of threads per physical core (useful for fine-tuning thread usage). |

---

### **6.7.3 Handling Multithreading in SLURM**
#### **Scenario 1: Running a Multithreaded Application**
For multithreaded applications, request the appropriate number of CPUs per task:

```bash
#!/bin/bash
#SBATCH --job-name=multithreaded_job
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1           # One task for the application
#SBATCH --cpus-per-task=16   # Use 16 CPUs (threads) for this task
#SBATCH --time=02:00:00
#SBATCH --output=multithreaded_%j.txt

module load some_multithreaded_module

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # OpenMP threads match allocated CPUs
srun ./your_multithreaded_application
```

**Explanation**:
- `--ntasks=1`: Single application instance.
- `--cpus-per-task=16`: Allocates 16 CPUs (threads) to the application.
- `OMP_NUM_THREADS`: Ensures the application uses the allocated CPUs effectively.

#### **Scenario 2: Explicitly Using Hyper-Threading**
If hyper-threading is enabled, you can allocate threads across logical CPUs:

```bash
#!/bin/bash
#SBATCH --job-name=ht_enabled_job
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80      # Use all logical CPUs (40 cores x 2 threads)
#SBATCH --hint=multithread      # Ensure hyper-threading is used
#SBATCH --time=01:00:00
#SBATCH --output=ht_enabled_%j.txt

module load some_module

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun ./your_multithreaded_application
```

---

### **6.7.4 Best Practices for Multithreading**
1. **Understand Your Application**:
   - Verify if your application benefits from multithreading. Some applications (e.g., OpenMP-based) perform better with carefully tuned thread counts.

2. **Test Hyper-Threading**:
   - Run tests to determine if hyper-threading improves or degrades performance. Use `--hint=multithread` or `--hint=nomultithread` accordingly.

3. **Match Threads to Resources**:
   - Use `--cpus-per-task` to match the number of threads your application will use. Over-requesting CPUs wastes resources and can block other users.

4. **Set Thread Environment Variables**:
   - For OpenMP applications, set `OMP_NUM_THREADS` to match `$SLURM_CPUS_PER_TASK`.
   - For other multithreaded libraries, such as Intel MKL, set variables like `MKL_NUM_THREADS`.

5. **Respect Shared Resources**:
   - Do not over-allocate CPUs or threads beyond your actual requirements. This ensures fair use of Cyclone's shared resources.

6. **Monitor Thread Usage**:
   - Use `scontrol show job <job_id>` or `squeue -j <job_id>` to verify that the requested resources match the allocation.
   - 
---

### **6.7.5 Hyper-Threading vs. No Hyper-Threading: Which to Use?**
| Use Case                                         | Recommendation                                                               |
| ------------------------------------------------ | ---------------------------------------------------------------------------- |
| **I/O-Bound Applications**                       | Enable hyper-threading (`--hint=multithread`) to maximize throughput.        |
| **CPU-Bound Applications**                       | Disable hyper-threading (`--hint=nomultithread`) for consistent performance. |
| **Memory-Intensive Applications**                | Disable hyper-threading to avoid resource contention.                        |
| **Highly Multithreaded Applications (e.g., ML)** | Enable hyper-threading to leverage additional logical CPUs.                  |

---

### 6.8 Using SLURM for GPU, MPI, and Hybrid Applications on Cyclone

Cyclone's partitions support diverse workloads, including GPU-accelerated applications, MPI-distributed applications, and hybrid MPI+X (MPI combined with multithreading or GPUs) setups. Each type of workload requires specific SLURM configurations to optimize resource usage and maximize performance.

---

### **6.8.1 Using GPUs with SLURM**
GPU resources are requested and allocated through SLURM’s `--gres` directive. This setup is suitable for GPU-only applications where one or more GPUs per node perform the bulk of the computation.

#### Example SLURM Script for a GPU Application
```bash
#!/bin/bash
#SBATCH --job-name=gpu_job
#SBATCH --partition=gpu           # The GPU partition must be used
#SBATCH --nodes=1
#SBATCH --gres=gpu:2              # Request 2 GPUs
#SBATCH --cpus-per-task=20        # Allocate 20 CPU cores (10 for each GPU)
#SBATCH --time=01:30:00
#SBATCH --account=<your-account>  # Project account to be charged

module load foss/2022a CUDA/12.1.1

export OMP_NUM_THREADS=20         # Use 20 threads

./gpu_application
```

#### Best Practises for GPU allocations:
1. **GPU Allocation**: Use `--gres=gpu:<count>` to specify the number of GPUs required per node.
2. **CPU-GPU Balance**: Allocate sufficient CPU cores to manage GPU operations effectively. A typical ratio is 5-10 CPU cores per GPU.
3. **Partition**: Use the `gpu` partition for GPU-enabled nodes.

---

### **6.8.2 Using MPI Applications with SLURM**
MPI applications distribute workloads across multiple nodes and/or cores. Cyclone’s interconnect and partitioning allow efficient scaling of MPI-based computations.

#### Example SLURM Script for MPI Applications
```bash
#!/bin/bash
#SBATCH --job-name=mpi_job
#SBATCH --partition=cpu
#SBATCH --nodes=4                 # Request 4 nodes
#SBATCH --ntasks-per-node=20      # Allocate 20 MPI ranks per node
#SBATCH --cpus-per-task=1         # One CPU core per MPI rank
#SBATCH --time=02:00:00
#SBATCH --account=<your-account>  # Project account to be charged

# Load the OpenMPI Library
module load OpenMPI/4.1.6-GCC-13.2.0

# Launch the application in parallel (used in place of mpirun)
srun ./mpi_application
```

#### Best Practises:
1. **Nodes and Ranks**: Use `--nodes` and `--ntasks-per-node` to control the number of MPI ranks and nodes.
2. **Partition**: Use the `cpu` partition for MPI jobs unless the application uses GPUs.
3. **Scaling**: Start small and scale based on application performance to ensure efficient resource usage.

---

### **6.8.3 Using Hybrid MPI+X Applications with SLURM**
Hybrid applications combine MPI for inter-node communication with threading (e.g., OpenMP) or GPU acceleration for intra-node parallelism. Cyclone supports such setups, requiring careful allocation of CPUs, threads, and GPUs.

#### Example SLURM Script for Hybrid MPI+X Applications
```bash
#!/bin/bash
#SBATCH --job-name=hybrid_job
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2       # 2 MPI ranks per node
#SBATCH --gres=gpu:2              # 2 GPUs per node
#SBATCH --cpus-per-task=10        # 10 CPU cores per MPI rank
#SBATCH --hint=nomultithread      # Use physical cores only
#SBATCH --time=03:00:00
#SBATCH --account=<your-account>  # Project account to be charged

module load OpenMPI/4.1.6-GCC-13.2.0 CUDA/12.1.1

export OMP_NUM_THREADS=10         # 10 threads per MPI rank
srun ./hybrid_application
```

**Important Note**: The `srun` command here will make sure that during execution 4 processes are spawned (2 on each node). Each process, will be responsible for 10 CPU threads and one GPU.

#### Example SLURM Script for Hybrid MPI+GPU Applications (Using GPU-GPU Communication)
```bash
#!/bin/bash
#SBATCH --job-name=hybrid_job
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2       # 2 MPI ranks per node
#SBATCH --gres=gpu:2              # 2 GPUs per node
#SBATCH --cpus-per-task=10        # 10 CPU cores per MPI rank
#SBATCH --hint=nomultithread      # Use physical cores only
#SBATCH --time=03:00:00
#SBATCH --account=<your-account>  # Project account to be charged

# Use CUDA-Aware MPI to enable NVLink GPU-GPU communication
module load OpenMPI/4.1.4-NVHPC-22.7-CUDA-11.7.0

export OMP_NUM_THREADS=10         # 10 threads per MPI rank
srun ./hybrid_application
```

**Important Note**: Cyclone supports NVLink connection between GPUs on the same node. This means that it is possible for GPUs to talk to each other by-passing the CPU, allowing for much higher bandwidth during communication. This can be enabled by using the `OpenMPI/4.1.4-NVHPC-22.7-CUDA-11.7.0` module, which loads a CUDA-Aware MPI version of the OpenMPI library.

#### Best Practises:
1. **CPU and Thread Management**:
   - Allocate CPU cores per rank using `--cpus-per-task`.
   - Set the number of threads using environment variables (e.g., `OMP_NUM_THREADS`).
   - Use `--hint=nomultithread` to avoid hyper-threading if required.
2. **GPU Usage**:
   - Distribute GPUs among ranks using `--gres=gpu:<count>` and ensure your application supports this configuration.
3. **Partition**: Use the `gpu` partition for hybrid workloads that require GPUs.

---