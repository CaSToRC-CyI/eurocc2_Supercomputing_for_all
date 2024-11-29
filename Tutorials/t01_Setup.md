<!--
 t01_Setup.md

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

# **Tutorial 1: Setup SSH Access, File Transfer, and Optional VS Code Installation**
## **1.1 Objective:**
- Enable SSH access to Cyclone.
- Transfer source files to Cyclone using `scp` or `rsync`.
- (Optional) Install and configure VS Code for remote development.

## **1.2 Workflow:**
### **1.2.1 SSH Access Setup:**
   - Follow instructions from **[Guide 3: Cyclone File Transfer](../Guides/g03_Cyclone_File%20Transfer.md)** to set up SSH access.
   - Verify access with:
     ```bash
     ssh your_username@cyclone.hpcf.cyi.ac.cy
     ```

---

### **1.2.2 File Transfer:**
   - Copy source files (`Code/`) from your local machine to Cyclone using scp:
     ```bash
     scp -r Code/ your_username@cyclone.hpcf.cyi.ac.cy:/nvme/h/your_username/
     ```
   - Verify the files on Cyclone:
     ```bash
     ls /nvme/h/your_username/Code/
     ```

---

### **1.2.3 (Optional) Install VS Code:**
#### **Step1: Install VS Code
   1. Download Visual Studio Code
     - Visit the [VS Code official website](https://code.visualstudio.com/).
     - Download the appropriate version for your operating system (Windows, macOS, or Linux).
   2. Install VS Code on your local machine.
     - **Windows**: Run the downloaded `.exe` file and follow the installation wizard.
     - **macOS**: Open the downloaded `.dmg` file and drag VS Code into the Applications folder.
     - **Linux**: Follow the installation instructions provided on the website, such as using a package manager (e.g., `sudo apt install code` for Ubuntu).
  
#### **Step 2: Install the Remote - SSH Extension**
  1. **Open Extensions**: Launch VS Code and click on the **Extensions** icon in the Activity Bar on the left side of the window (looks like a square with four smaller squares).
  2. **Search for Remote - SSH**: In the search bar at the top of the Extensions view, type `Remote - SSH`.
  3. **Install the Extension**: Click on the **Install** button for the "Remote - SSH" extension by Microsoft.
   
#### **Step 3: Configure Remote SSH**

If you have already created a Config file as shown in **[Guide 2: Cyclone Accessing](../Guides/g02_Cyclone_Accessing.md#24-configuring-ssh-with-a-config-file)**, skip to 4.

1. **Open the Command Palette**:
   - Press `Ctrl + Shift + P` (Windows/Linux) or `Cmd + Shift + P` (macOS) to open the Command Palette.

2. **Add New SSH Host**:
   - In the Command Palette, type `Remote-SSH: Add New SSH Host` and select it.
   - Enter the SSH command to connect to Cyclone:
     ```bash
     ssh your_username@cyclone.hpcf.cyi.ac.cy
     ```
   - Choose the SSH configuration file to update:
     - Default: `~/.ssh/config`.

3. **Verify Configuration**:
   - Open the `~/.ssh/config` file to ensure the Cyclone entry exists:
     ```text
     Host cyclone
         HostName cyclone.hpcf.cyi.ac.cy
         User your_username
         IdentityFile ~/.ssh/id_rsa
     ```

4. **Connect to Cyclone**:
   - In the Command Palette, type `Remote-SSH: Connect to Host` and select it.
   - Choose `cyclone` from the list of configured hosts.

5. **Authenticate**:
   - If prompted, enter the passphrase for your private key or your system password.

6. **Verify Connection**:
   - Once connected, you will see the VS Code workspace loaded for Cyclone. You can open folders and edit files directly on the remote system.

#### **Optional: Install Recommended Extensions for Remote Development**
1. **Python Extension** (for Python scripts): Search for and install the "Python" extension by Microsoft.
2. **C/C++ Extension** (for C/C++ development): Search for and install the "C/C++" extension by Microsoft.

3. **Jupyter Extension** (for Jupyter notebooks): Search for and install the "Jupyter" extension by Microsoft.

4. **Other Tools**: Install any additional extensions based on your workflow and programming needs.

**Important Note**: Any package you are installing to be used during Remote Development will be executed on the login nodes of Cyclone. It is important therefore to avoid installing heavy tools as these will overload the login node and affect everyone that is using it.

---