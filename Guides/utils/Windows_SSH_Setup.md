# SSH Manager Installation Guide for Windows
## OpenSSH
### 1. Check if OpenSSH Client is installed:
1. Open a Windows PowerShell as Administrator.
   - **Press Windows Key > Search "Windows PowerShell" > Right-click "Run as Administrator"**
2. Run the following command to see if the OpenSSH client is installed:
    ```powershell
    Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Client*'
    ```
    If installed, you should see something like:
    ```powershell
    Name: OpenSSH.Client~~~~0.0.1.0
    State: Installed
    ```

### 2. Install OpenSSH Client (If not already Installed!)
1. Open Settings
2. Search for OpenSSH Client:
   - Click on **Add a feature** and search for "OpenSSH Client"
   - Select it and click **Install**.

### 3. Activate OpenSSH (Start Services)
1. Open a Windows PowerShell as Administrator.
   - **Press Windows Key > Search "Windows PowerShell" > Right-click "Run as Administrator"**
2. Start the SSH Agent:
   ```powershell
    Start-Service ssh-agent
   ```

### WSL
1. Open a Windows PowerShell as Administrator.
   - **Press Windows Key > Search "Windows PowerShell" > Right-click "Run as Administrator"**
2. For Windows 10 (build 19041 and higher) and Windows 11, you can use the simplified command to install WSL:
    ```powershell
    wsl --install
    ```

    This will:
    - Enable the WSL feature.
    - Install the default Linux distribution (usually Ubuntu).
    - Install the necessary Virtual Machine Platform and Windows Subsystem for Linux components.
3. Restart Your Computer:
If prompted, restart your computer to complete the installation.

## Git Bash
1. Install Git Bash:
   - Install Git for Windows if you haven’t already [Download Git](https://git-scm.com/downloads).
2. Launch Git Bash
   - **Press Windows Key > Search "Git Bash"**