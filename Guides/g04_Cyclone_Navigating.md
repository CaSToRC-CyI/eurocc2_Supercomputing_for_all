<!--
 g04_Cyclone_Navigating.md

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

---

# **4. Navigating the Cyclone Directory Structure**

When working on Cyclone, understanding its directory structure and knowing how to navigate it are essential for efficient usage. Cyclone's directory system includes specific areas optimized for different types of data storage, with unique performance characteristics and retention policies. Please refer to [Cyclone Directory Structure Guide](g03_Cyclone_File%20Transfer.md#32-cyclone-directory-structure) for more information.

---

## **4.1 Cheat Sheet: Basic Navigation Commands**
This cheat sheet serves as a quick reference for frequently used navigation and file management commands, along with examples to illustrate their usage. Each command is described in the following sections.

| **Command**                    | **Description**                                      | **Example**                                                           |
| ------------------------------ | ---------------------------------------------------- | --------------------------------------------------------------------- |
| `pwd`                          | Show the current working directory.                  | `pwd`                                                                 |
| `ls`                           | List files and directories in the current location.  | `ls -lah` (detailed and human-readable sizes, including hidden files) |
| `cd /path/to/directory`        | Change to the specified directory.                   | `cd /nvme/scratch/your_username/`                                     |
| `cd ~`                         | Navigate to your home directory.                     | `cd ~`                                                                |
| `cd ..`                        | Move to the parent directory.                        | `cd ..`                                                               |
| `mkdir /path/to/new/directory` | Create a new directory at the specified location.    | `mkdir /nvme/scratch/your_username/new_project`                       |
| `cp source_path destination`   | Copy files or directories. Use `-r` for directories. | `cp simulation_output.dat /nvme/h/your_username/`                     |
| `mv source_path destination`   | Move or rename a file or directory.                  | `mv temp_files /nvme/h/your_username/my_temp_files`                   |
| `rm file_name`                 | Delete a file. Use `-r` to remove a directory.       | `rm -r /nvme/scratch/your_username/old_simulation`                    |
| `head file_name`               | Display the first few lines of a file.               | `head simulation_output.dat`                                          |
| `tail file_name`               | Display the last few lines of a file.                | `tail -n 20 simulation_output.dat`                                    |
| `less file_name`               | View file contents with scrolling.                   | `less simulation_output.dat`                                          |
| `tar -czvf archive.tar.gz dir` | Compress a directory into a `.tar.gz` file.          | `tar -czvf results.tar.gz /nvme/scratch/your_username/simulation_run` |
| `tar -xzvf archive.tar.gz`     | Extract files from a `.tar.gz` archive.              | `tar -xzvf results.tar.gz`                                            |
| `du -sh /path/to/dir`          | Check the size of a directory or file.               | `du -sh /nvme/scratch/your_username/simulation_output`                |
| `find /path -name "pattern"`   | Search for files or directories matching a pattern.  | `find /nvme/h/your_username/ -name "*.dat"`                           |

---

## **4.2 Basic Navigation Commands**

Here are essential commands for navigating Cyclone's directory structure:

### **4.2.1 Check Your Current Directory**
```bash
pwd
```
This command displays the current working directory. For example, upon logging in on Cyclone, the `pwd` command will result in:
```bash
$ pwd
/nvme/h/your_username
```

---

### **4.2.2 List Files and Directories**
```bash
ls
```
Lists the contents of the current directory. Add options for more details:
- `ls -l`: Long format, shows file sizes, permissions, and timestamps.
- `ls -a`: Includes hidden files (e.g., `.ssh`).
- `ls -lh`: Human-readable file sizes.

Example:
```bash
$ ls -lh /nvme/scratch/your_username/
total 5.2G
-rw-r--r-- 1 user group  1.2G Nov 25 12:00 simulation_output1.dat
-rw-r--r-- 1 user group  2.3G Nov 25 12:15 simulation_output2.dat
```

---

### **4.2.3 Change Directory**
```bash
cd /path/to/directory
```
Moves to the specified directory. Common shortcuts:
- `cd ~`: Home directory.
- `cd ..`: Parent directory.
- `cd -`: Switch to the previous directory.

Example:
```bash
$ cd /nvme/scratch/your_username/
$ pwd
/nvme/scratch/your_username/
$ cd ..
$ pwd
/nvme/scratch/
```

---

### **4.2.4 Create a New Directory**
```bash
mkdir /path/to/new/directory
```
Creates a new directory at the specified path. 

Example:
  ```bash
  $ mkdir /nvme/scratch/your_username/temp_files
  ```

#### **Troubleshooting**
Assuming my current directory is `/nvme/scratch/your_username`, and I want to create `/nvme/scratch/your_username/project1/results`. In the case where the path `/nvme/scratch/your_username/project1` doesn't exist, `mkdir` will result in the following error:
  ```bash
  $ pwd
  /nvme/scratch/your_username/
  $ mkdir /nvme/scratch/your_username/project1/results
  mkdir: cannot create directory ‘/nvme/scratch/your_username/project1’: No such file or directory
  ```
Using the `-p` option ensures all intermediate directories are created:
  ```bash
  $ mkdir -p /nvme/scratch/your_username/project1/results
  ```
This command creates:
1. `/nvme/scratch/your_username/project1/`, if it doesn’t exist.
2. `/nvme/scratch/your_username/project1/results/`.
---

### **4.2.5 Copy Files and Directories**
```bash
cp source_path destination_path
```
Copies files or directories.
- Use `-r` to copy directories recursively.

Example:
```bash
$ cp /nvme/scratch/your_username/simulation_output1.dat /nvme/h/your_username/
```

---

### **4.2.6 Move or Rename Files**
```bash
mv source_path destination_path
```
Moves a file or directory to a new location or renames it.

Example:
```bash
$ mv /nvme/scratch/your_username/temp_files /nvme/h/your_username/my_temp_files
```

---

### **4.2.7 Remove Files and Directories**
```bash
rm file_name
```
Deletes a file. Use `-r` for directories.

Example:
```bash
$ rm /nvme/scratch/your_username/simulation_output1.dat
$ rm -r /nvme/scratch/your_username/temp_files
```

---

### **4.2.8 View File Contents**
- **Display the first few lines**:
  ```bash
  head file_name
  ```
- **Display the last few lines**:
  ```bash
  tail file_name
  ```
- **View entire file with scrolling**:
  ```bash
  less file_name
  ```

Example:
```bash
$ head simulation_output1.dat
```

---

## **4.3 Example Scenario: Running a Simple Bash Script**

In this scenario, we’ll create a simple bash script to run on the login node that will display its hostname. The script will be created in the user's Home directory. To practise, the directory will the be copied to the Scratch directory, from where the script will be executed. The steps include creating and copying a directory with it's contents, creating a very simple script, making it executable, and running it.

#### **Step 1: Create the Example Directory**
```bash
cd $HOME
mkdir -p tutorials/example_4_3
cd tutorials/example_4_3
```

#### **Step 2: Create the Bash Script**
Use the `echo` command with redirection to create a bash file named `hello_node.sh`:
```bash
echo '#!/bin/bash' > hello_node.sh
echo 'echo "Hello from Node $HOSTNAME"' >> hello_node.sh
```

#### **Step 3: Change the File Permission**
Make the script executable:
```bash
chmod +x hello_node.sh
```

#### **Step 4: Copy directory in User Scratch**
Copy the example directory and it's contents in the user scratch:
```bash
cd $HOME
cp -R tutorials/example_4_3 ~/scratch
```
To check the copy was successfull:
```bash
ls -la ~/scratch/example_4_3
total 2
drwxr-xr-x 2 user group   1 Nov 28  22:40 .
drwxr-x--- 5 user group   6 Nov 28  22:40 ..
-rwxr-xr-x 1 user group   45 Nov 28 22:40 hello_node.sh
```

#### **Step 5: Run the Script**
Execute the script:
```bash
cd ~/scratch/example_4_3
./hello_node.sh
```

#### **Expected Output**
The script will display:
```bash
Hello from Node front02
```
where `front02` indicates that the script was executed on the login node.

---

### **Clarifications**
1. **Creating the Script Without Opening a File**:
   - `echo '#!/bin/bash' > hello_node.sh`: Writes the shebang to indicate it’s a bash script.
   - `echo 'echo "Hello from Node $HOSTNAME"' >> hello_node.sh`: Appends the echo command to the script.

2. **Changing Permissions**:
   - `chmod +x hello_node.sh`: Ensures the script is executable.

3. **Running the Script**:
   - `./hello_node.sh`: Executes the script from the current directory.
§
---

## **Important Notes**
- **No Backups**: None of the directories on Cyclone are backed up. It is your responsibility to save important files elsewhere.
- **Retention Policies**: Files in the scratch directory may be deleted to free up space, so regularly move important files to your home directory or local storage.
- **Quota Management**: Be mindful of your storage usage and quotas in shared directories.

---