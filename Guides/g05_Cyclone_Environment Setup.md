<!--
 g05_Cyclone_Environment Setup.md

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

# 5. Cyclone Environment Setup Guide

This guide provides users with the knowledge and skills to effectively navigate and utilize Cyclone's module system, set up a local Conda environment, and manage software dependencies for their projects.

---

## **5.1 Understanding the Module System**

Cyclone uses the **Lmod** module system, which allows users to dynamically manage their software environment. Modules load or unload specific software versions, ensuring compatibility and preventing conflicts between dependencies.

### **Basic Commands**

| Command                  | Description                        |
| ------------------------ | ---------------------------------- |
| `module avail`           | Lists all available modules.       |
| `module load <module>`   | Loads the specified module.        |
| `module unload <module>` | Unloads the specified module.      |
| `module list`            | Displays currently loaded modules. |
| `module purge`           | Unloads all loaded modules.        |

### **Example: Loading Python**
1. By default once connected to Cyclone, no modules are loaded:
   ```bash
   module list
      No modules loaded
   ```
2. Check available Python modules:
   ```bash
   module avail Python
   ```
   will present various Python versions:
   ```bash
   ---------------------- /eb/modules/all ----------------------
   ...
   Python/2.7.16-GCCcore-8.3.0             
   Python/2.7.18-GCCcore-11.2.0                        
   Python/3.7.4-GCCcore-8.3.0                 
   Python/3.8.6-GCCcore-10.2.0                       
   Python/3.9.6-GCCcore-11.2.0           
   Python/3.10.4-GCCcore-11.3.0
   ...
   Where:
   D:  Default Module

   If the avail list is too long consider trying:

   "module --default avail" or "ml -d av" to just list the default modules.
   "module overview" or "ml ov" to display the number of modules for each name.

   Use "module spider" to find all possible modules and extensions.
   Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".
   ```

3. Load a specific version:
   The default Python version on Cyclone can be displayed by:
   ```bash
   python --version
      Python 3.10.13
   ```
   The default version can be changed by loading a different module:
   ```bash
   module load Python/3.9.6-GCCcore-11.2.0 
   ```
   Now the Python version is:
   ```bash
   python --version
      Python 3.9.6
   ```
4. Verify the loaded module:
   ```bash
   module list
   Currently Loaded Modules:
      1)  GCCcore/11.2.0
      2)  zlib/1.2.11-GCCcore-11.2.0
      3)  binutils/2.37-GCCcore-11.2.0
      4)  bzip2/1.0.8-GCCcore-11.2.0
      5)  ncurses/6.2-GCCcore-11.2.0
      6)  libreadline/8.1-GCCcore-11.2.0
      7)  Tcl/8.6.11-GCCcore-11.2.0
      8)  SQLite/3.36-GCCcore-11.2.0 
      9)  XZ/5.2.5-GCCcore-11.2.0
      10) GMP/6.2.1-GCCcore-11.2.0
      11) libffi/3.4.2-GCCcore-11.2.0  
      12) OpenSSL/1.1  
      13) Python/3.9.6-GCCcore-11.2.0
   ```
   **Important Note:** The module system loaded the requested Python module and the required dependencies for that module to be functional.
   
---

## **5.2 Setting Up a Local Conda Environment**

Conda is a powerful tool for managing Python environments and packages. Cyclone provides pre-installed Anaconda modules, or you can install Conda in your home directory.

### **Step 1: Load Anaconda Module**
Load the latest Anaconda module:
```bash
module load Anaconda3/2023.03-1
```

Prior to Anaconda being loaded, the terminal looked like:
   ```bash
   [cstyl@front02 ~]$
   ```
Once Anaconda is loaded, the terminal now looks like:
   ```bash
   (base)[cstyl@front02 ~]$
   ```
indicating the `base` conda environment is loaded. 

**Important Note:** This environment provides pre-installed Anaconda modules and can be used by all users. However, users cannot install their own packages in it.

### **Step 2: Create a New Environment**
Create an isolated Conda environment named `myenv`:
```bash
conda create --name myenv python=3.10
```

### **Step 3: Activate the Environment**
Activate the environment to start working with it:
```bash
conda activate myenv
```
Once the environment is activated, the terminal now looks as:
   ```bash
   (myenv) [cstyl@front02 ~]$
   ```
**Important Note:** This environment is now empty and only available to the user created it. This environment lives on user's home directory and the user can install their own packages.

### **Step 4: Install Packages**
Install additional packages, such as NumPy and SciPy:
```bash
conda install numpy scipy
```

### **Step 5: Deactivate the Environment**
When done, deactivate the environment:
```bash
conda deactivate
```

---

## **5.3 Example Scenario: Running a Python Script with Conda**

### **Step 1: Load the Anaconda module and activate your Conda environment**:
   ```bash
   module load Anaconda3/2023.03-1
   conda activate myenv
   ```

### **Step 2: Create a Python script**:
   Use the `echo` command to create a Python script named `version_info.py` without opening a text editor:
   ```bash
   mkdir -p $HOME/tutorials/example_5_3
   cd $HOME/tutorials/example_5_3
   echo 'import numpy as np' > version_info.py
   echo 'import scipy' >> version_info.py
   echo 'import sys' >> version_info.py
   echo 'print(f"Numpy Version: {np.__version__}")' >> version_info.py
   echo 'print(f"SciPy Version: {scipy.__version__}")' >> version_info.py
   echo 'print(f"Python Version: {sys.version}")' >> version_info.py
   ```

### **Step 3: Run the Python script**:
   Execute the script to display the versions of NumPy and Python:
   ```bash
   python version_info.py
   ```

### **Step 4: Expected Output**:
   The script will print the installed versions of NumPy and Python. For example:
   ```
   Numpy Version: 2.0.0
   SciPy Version: 1.14.1
   Python Version: 3.10.15 (main, Oct  3 2024, 07:27:34) [GCC 11.2.0]
   ```

This example ensures that your environment is set up correctly and verifies the versions of critical dependencies before proceeding with further development.

---

## **5.4 Best Practices**
1. **Always Load Required Modules**: Ensure you load all dependencies before running your software.
2. **Purge Modules When Switching Projects**: Use `module purge` to avoid conflicts.
3. **Keep Your Conda Environments Clean**: Create separate environments for different projects to maintain organization and avoid dependency clashes.
4. If you frequently use the same modules, you can save your configuration for easier loading:
   ```bash
   module save my_session
   ```
   and restore it later with:
   ```bash
   module restore my_session
   ```
5. Some modules may require others to function correctly. To check module descriptions you can run:
   ```bash
   module show <module_name>
   ```
   Note however that depending on the system, the dependencies might be handled by the module system itself.
6. For reproducibility (or backup), document the installed Conda packages:
   - Export the list of installed packages:
      ```bash
      conda list --export > requirements.txt
      ```
   - Then, to recreate the environment, use:
      ```bash
      conda create --name my_project_env --file requirements.txt
      ```

---