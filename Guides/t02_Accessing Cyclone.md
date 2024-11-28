# 2. Accessing Cyclone
<div style="text-align: justify;">

This guide is designed to help new users access and utilize the **Cyclone** HPC system securely and efficiently. It provides clear, step-by-step instructions for generating SSH keys, configuring your system for seamless access, and connecting to Cyclone from **Mac**, **Linux**, and **Windows** platforms. **SSH (Secure Shell)** is a vital tool for accessing remote systems like Cyclone, ensuring your data and credentials remain secure while you leverage the powerful computing resources available. Whether you are a first-time user or need a refresher, this guide will walk you through the setup process and help you get started with confidence.
</div>

## **What is SSH and Why is it Important?**
<div style="text-align: justify;">
SSH, or Secure Shell, is a secure way to access and manage remote systems, such as High-Performance Computing (HPC) resources, over a network. It encrypts all communication, protecting sensitive information from being intercepted by unauthorized users. SSH is essential because it provides a safe and efficient way to connect to powerful remote systems for tasks like running simulations, managing files, and analyzing data. Instead of using vulnerable passwords, SSH often uses a system called public-key cryptography to verify your identity.

Here’s how it works: SSH relies on a pair of keys—a **public key** and a **private key**. The public key is shared with the remote system (the server), acting like a lock, while the private key stays safely on your computer, working as the unique key that can open that lock. When you try to connect, the server sends a challenge that only your private key can solve. If it’s solved correctly, the server knows it’s you, and the connection is established securely. This approach ensures that even if someone intercepts the communication, they can’t access your data or impersonate you. SSH combines simplicity and robust security, making it an indispensable tool for accessing and using HPC systems effectively.
</div>

## **2.1. Generating a New SSH Key**
<div style="text-align: justify;">
The first step in getting access on an HPC machine is to create a pair of SSH keys. The public key is shared with the remote system (the server), acting like a lock, while the private key stays safely on your computer, working as the unique key that can open that lock.

Below we provide instructions on how to generate the pair of keys depending on the Operating System (Mac, Linux or Windows) the user uses.
</div>

### **2.1.1 Mac/Linux**
1. Open a terminal.
2. Generate a new SSH key pair:
    ```bash
    ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
    ```
   - Press Enter to save the key to the default location (`~/.ssh/id_rsa`).
   - Set a passphrase for added security (optional).

3. Add the SSH key to the agent:
    ```bash
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_rsa
    ```

### **2.1.2 Windows**
Note that the following options assume the equivalent tool is already installed in the system. Additionally, each option might require **Administrative Priviledges**.

If you need to install any of the options below, look at the [Installation Guide](utils/Windows_SSH_Setup.md)

#### **Option 1: Using OpenSSH via PowerShell**
1. Open PowerShell.
   - **Press Windows Key > Search "Windows PowerShell" > Enter**
2. Generate a new SSH key pair:
    ```powershell
    ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
    ```
   - Press Enter to save the key to the default location (`~/.ssh/id_rsa`).
   - Set a passphrase for added security (optional).

3. Add the SSH key to the agent:
    ```powershell
    Start-Service ssh-agent
    ssh-add ~\.ssh\id_rsa
    ```

#### **Option 2: Using WSL**
1. Open a WSL terminal.
2. Generate a new SSH key pair:
    ```bash
    ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
    ```
   - Press Enter to save the key to the default location (`~/.ssh/id_rsa`).
   - Set a passphrase for added security (optional).

3. Add the SSH key to the agent:
    ```bash
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_rsa
    ```

#### **Option 3: Using Git Bash**
1. Download and install [PuTTY](https://www.putty.org/) and PuTTYgen.
2. Open PuTTYgen.
   - Select **RSA** and set the key size to 4096 bits.
   - Click **Generate** and move the mouse around the blank area to generate randomness.
   - Save the private key (`.ppk`) and public key.
3. Use the `.ppk` file with PuTTY to connect to the HPC system.

---

## **2.2. Connecting to the HPC System via SSH**

1. Open a terminal or PowerShell (or WSL, Git Bash).
2. Use the following command, replacing placeholders:
   ```bash
   ssh <your_username>@<HPC address>
   ```
   In the case of Cyclone (`cyclone.hpcf.cyi.ac.cy`), assuming I am user `cstyl` the command looks as follows: 
   ```bash
   ssh cstyl@cyclone.hpcf.cyi.ac.cy
   ```

    Upon successful access to the system, you will see the following message being displayed:
   ![HPC System](images/cyclone_welcome.png)
---

## **2.3. SSH with a Config File**
<div style="text-align: justify;">
An SSH config file simplifies and streamlines the process of connecting to remote systems by allowing you to save connection settings like the hostname, port, username, and identity file. This eliminates the need to repeatedly type lengthy commands, making it especially useful when managing multiple servers. By assigning nicknames (aliases) for connections, the config file makes switching between servers quick and error-free. It also enhances security by centralizing configurations and reducing the risk of mistakes, while saving time for repeated tasks. 
</div>

### **Mac/Linux**
1. Open the SSH configuration file or create one if it doesn't exist:
   ```bash
   nano ~/.ssh/config
   ```
2. Add the following configuration, replacing placeholders with your details:
   ```
   Host my-hpc
       HostName <HPC address>
       User <your_username>
       IdentityFile ~/.ssh/id_rsa
   ```
    For example, assuming I (`cstyl`) want to access Cyclone (`cyclone.hpcf.cyi.ac.cy`) with my private key (`~/.ssh/id_rsa`), then the entry in the config file will look like as follows:
    
   ```bash
   Host cyclone
       HostName cyclone.hpcf.cyi.ac.cy
       User cstyl
       IdentityFile ~/.ssh/id_rsa
   ```
3. Save and exit the editor.
4. Now, in order to connect to Cyclone you can do so by running the following command:
   ```bash
   ssh cyclone
   ```
### **Windows**

#### **Using PowerShell, WSL or Git Bash**
1. Navigate to the `.ssh` directory:
   ```powershell
   cd ~\.ssh
   ```
   or
   ```bash
   cd ~/.ssh
   ```
2. Create or edit the config file:
   ```powershell
   notepad config
   ```
3. Add the configuration details (same format as Mac/Linux).
4. Save the file. **Note** that the file must have no extension (e.g., `.txt`). In case the file is saved with an extension

---

## **2.4. Notes and Troubleshooting**

### **2.4.1 `~/.ssh` directory does not exist on Windows**
**Option 1:** Create the direcotry using Powershell
1. Open PowerShell.
   - **Press Windows Key > Search "Windows PowerShell" > Enter**
2. Navigate to your Home directory
    ```powershell
    cd ~
    ```
    This will take you to your home directory, typically something like `C:\Users\<YourUsername>`.
3. Create the `.ssh` directory:
   ```powershell
    mkdir .ssh
   ```
4. Verify that the directory was created:
   ```powershell
    ls .ssh
   ```
   If the `.ssh` folder exists, the command will list its contents (it may be empty if just created).

**Option 2:**
1. Open File Explorer.
2. Navigate to your home directory: `C:\Users\<YourUsername>`.
3. Create a new folder named `.ssh`:
    - Right-click and choose **New > Folder**.
    - Name it `.ssh` (include the period).
    - Confirm if prompted about using a name that starts with a period.

### **2.4.2. Changing and Removing File Extensions (Windows)**
File extensions (like `.txt`, `.png`, `.exe`) are often hidden by default in Windows, so you'll first need to make extensions visible before removing or changing them.
1. Open **File Explorer**.
   - Press **Win + E** or click the folder icon in the taskbar.
2. Access View Options:
    - **Windows 10**: Click on the View tab in the toolbar at the top.
    - **Windows 11**: Click on the three dots (`...`) in the toolbar at the top and choose **Options**.
3. Click on the tab **View**, go to *"Advanced settings"* and uncheck the checkbox *"Hide extensions for known file types"* if already checked.
4. (Now that the extension is visible) Rename the file:
   - Right-click the file and choose **Rename**.
   - Remove or modify the extension as needed.
   - Confirm the change when prompted.

### **2.4.2. Show/Unhide `.ssh` directory (Windows)**
File extensions (like `.txt`, `.png`, `.exe`) are often hidden by default in Windows, so you'll first need to make extensions visible before removing or changing them.
1. Open **File Explorer**.
   - Press **Win + E** or click the folder icon in the taskbar.
2. Access View Options:
    - **Windows 10**: Click on the View tab in the toolbar at the top.
    - **Windows 11**: Click on the three dots (`...`) in the toolbar at the top and choose **Options**.
3. Click on the tab **View**, go to *"Advanced settings"*.
4. Scroll down to *"Hidden files and folders"* and select the option **Show hidden files, folders, and drives**.
---