<!--
 t03_C_Cpp.md

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

# **Tutorial 3: Running C on Compute Nodes**
## **3.1 Objective**

The primary goal of **Tutorial 3** is to equip users with the knowledge and skills to compile and execute various C/C++ applications effectively on Cyclone's compute nodes. By leveraging Cyclone's high-performance infrastructure, users will explore the process of running simple "Hello World" programs across multiple paradigms, including **serial execution**, **parallel computing with OpenMP**, **distributed computing with MPI**, and **GPU-accelerated computing with CUDA**. This hands-on experience highlights the nuances of using different compilers (GNU and Intel), and how to use the module system, understanding the role of SLURM scripts for job submission, and optimizing resource usage.

## **3.2 Workflow**
### **3.2.1 Prepare Source Files**

**Plain C:** Save the following as `hello.c`:
```c
#include <stdio.h>
int main() {
    printf("Hello, World from Cyclone!\n");
    return 0;
}
```

**OpenMP:** Save the following as `hello_openmp.c`:
```c
#include <stdio.h>
#include <omp.h>
int main() {
    #pragma omp parallel
    {
        printf("Hello from thread %d out of %d\n", omp_get_thread_num(), omp_get_num_threads());
    }
    return 0;
}
```

**MPI:** Save the following as `hello_mpi.c`:
```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("Hello from process %d out of %d\n", rank, size);
    MPI_Finalize();
    return 0;
}
```

**CUDA:** Save the following as `hello_cuda.cu`:
```cpp
#include <stdio.h>

__global__ void helloFromGPU() {
    printf("Hello, World from GPU thread %d\n", threadIdx.x);
}

int main() {
    helloFromGPU<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

### **3.2.2 Compile and Generate SLURM Scripts**
   Follow **[Guide 7: Compiling and Running C/C++ Code](../Guides/g07_Cyclone_Compiling.md)**.

### 3.2.3 Submit Jobs**
   Use `sbatch` to submit the generated SLURM scripts. In each submission add the `--reservation=edu25` to avoid queuing. For example:
   ```bash
    sbatch run_openmp.slurm --reservation=edu25
   ```

---