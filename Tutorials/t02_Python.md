<!--
 t02_Python.md

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

# **Tutorial 2: Running Python on Compute Nodes with Conda Environment Setup**
## 2.1 Objective
This tutorial focuses on training a simple AI model using PyTorch on Cyclone's compute nodes. It includes setting up a conda environment, writing Python code to train the model, and configuring SLURM scripts to run on either CPU or GPU nodes. Additionally, the script automatically detects GPU availability and adjusts accordingly.

---

## 2.2 Workflow
### **2.2.1 Load Anaconda and Create a Conda Environment**
1. **Log in to Cyclone**:
   ```bash
   ssh your_username@cyclone.hpcf.cyi.ac.cy
   ```

2. **Load the Anaconda Module**:
   ```bash
   module load Anaconda3/2023.03-1
   ```

3. **Create a Conda Environment**:
   ```bash
   conda create -n ai_env python=3.9 -y
   conda activate ai_env
   ```

4. **Install Required Libraries**:
   ```bash
   pip install torch torchvision matplotlib
   ```

---

### **2.2.2 Python Code for Training the AI Model**
Save the following code to a file named `train_model.py` in your home directory:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.fc(x)

# Generate synthetic data
torch.manual_seed(42)
x = torch.rand(100, 1).to(device)
y = 3 * x + torch.rand(100, 1).to(device)

# Model, loss, and optimizer
model = SimpleModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
epochs = 10
losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(x)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f"Epoch {epoch}, Loss: {loss.item()}")

# Plot the loss curve
plt.plot(range(epochs), losses, label='Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.savefig("loss_curve.png")
```

---

### **2.2.3 SLURM Scripts**

#### **For GPU Nodes**
Save the following script as `run_gpu.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=ai_train_gpu
#SBATCH --output=ai_train_gpu-%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --account=<your-account>

module load Anaconda3/2023.03-1
source ~/.bashrc
conda activate ai_env

python train_model.py
```

#### **For CPU Nodes**
Save the following script as `run_cpu.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=ai_train_cpu
#SBATCH --output=ai_train_cpu-%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --account=<your-account>

module load Anaconda3/2023.03-1
source ~/.bashrc
conda activate ai_env

python train_model.py
```

---

### **2.2.4 Submit the Job**
Submit the SLURM script:
```bash
sbatch run_gpu.slurm --reservation=edu25 # For GPU nodes
# or
sbatch run_cpu.slurm --reservation=edu25  # For CPU nodes
```

---

### **2.2.5 Displaying the Loss Plot**

After the job completes, a `loss_curve.png` file will be created in your working directory.

On your local machine, use `scp` to copy the image:
   ```bash
   scp your_username@cyclone.hpcf.cyi.ac.cy:/path/to/loss_curve.png .
   ```
View the image using your preferred image viewer.

---