# DiGress: Discrete Denoising diffusion models for graph generation

Update (Nov 20th, 2023): Working with large graphs (more than 100-200 nodes)? Consider using SparseDiff, a sparse version of DiGress: https://github.com/qym7/SparseDiff

Update (July 11th, 2023): the code now supports multi-gpu. Please update all libraries according to the instructions. 
All datasets should now download automatically

  - For the conditional generation experiments, check the `guidance` branch.
  - If you are training new models from scratch, we recommand to use the `fixed_bug` branch in which some neural
network layers have been fixed. The `fixed_bug` branch has not been evaluated, but should normally perform better.
If you train the `fixed_bug` branch on datasets provided in this code, we would be happy to know the results.

## Environment installation on Ubuntu (Runpod Pods)
This code was tested with PyTorch 2.0.1, cuda 11.8 and torch_geometrics 2.3.1

### Runpod Setup

  - Download anaconda/miniconda

    ```cd /workspace```

    ```wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh```

  - Run the Installation Script
  When running the script, you must manually specify the installation path.

  - Execute the script:```bash Miniconda3-latest-Linux-x86_64.sh```
  Review the license agreement (press Enter and then type `yes`).
  Crucial Step: When prompted for the installation location, it will default to `/root/miniconda3`. Do not press `enter`. Instead, type:`/workspace/miniconda3`.
  When asked if you want to initialize Miniconda by running conda init, type `yes`.

  - Apply Changes
  For the changes to take effect in your current terminal session, source your bash configuration: `source ~/.bashrc`

  - Verify and Clean Up
  Confirm that the installation is correctly mapped to the volume and remove the installer script to save space:
  `which conda`  
  Should return `/workspace/miniconda3/bin/conda`

    Also, remove the installer
  `rm Miniconda3-latest-Linux-x86_64.sh`

  - Set Up an Organized Directory Structure
  
    `mkdir -p /workspace/projects`

  - Clone via HTTPS

    ```cd /workspace/projects```
    ```git clone https://github.com/cvignac/DiGress.git```
    `cd DiGress`

  - Relink your existing installation after restart/migration:

    ```/workspace/miniconda3/bin/conda init bash && source ~/.bashrc```

  ### Python Setup

  - Create a rdkit environment that directly contains rdkit:
    
    ```conda create -c conda-forge -n digress rdkit=2023.03.2 python=3.9```
    `conda activate digress`
  - Check that this line does not return an error:
    
    ``` python3 -c 'from rdkit import Chem' ```
  - Install graph-tool (https://graph-tool.skewed.de/): 
    
    ```conda install -c conda-forge graph-tool=2.45```
  - Check that this line does not return an error:
    
    ```python3 -c 'import graph_tool as gt' ```
  - Install the nvcc drivers for your cuda version. For example:
    
    ```conda install -c "nvidia/label/cuda-11.8.0" cuda```

  - Uninstall the current pytorch: 
    
    ```pip uninstall torch torchvision torchaudio -y```
    ```pip uninstall torch-geometric -y```

  - Install a corresponding version of pytorch, for example: 
    
    ```pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118```

  - Install a compatible version of PyG, for example: 
    
    ```pip install torch-geometric==2.3.1```
    
  - Install other packages using the requirement file: 
    
    ```pip install -r requirements.txt```

  - Run:
    
    ```pip install -e .```

  - Navigate to the ./src/analysis/orca directory and compile orca.cpp: 
    
     ```g++ -O2 -std=c++11 -o orca orca.cpp```

Note: graph_tool and torch_geometric currently seem to conflict on MacOS, I have not solved this issue yet.

### Git Commit

  - Use a Personal Access Token (PAT)
  
    Go to your GitHub Settings > Developer Settings > Personal Access Tokens > Tokens (classic).
    Generate a new token with repo permissions. Allow: 
    
    `Permissions Category: Repository permissions`

    `Permission Name: Contents`

    `Level: Read and write`
    When you run git push, and it asks for your password, paste the token instead.

  - Configure your Identity

    ```git config --global user.email "your_email@example.com"```
    ```git config --global user.name "Your Name"```

  - Stage and Commit your changes

    ```git add src/datasets/inpatient_dataset.py```
    ```git commit -m "Add inpatient_dataset.py to /datasets/"```

  - Update the Remote URL with your Token

    ```git remote set-url origin https://<digress-runpod-access-token>@github.com/Haoheng-Tang/DiGress.git```

  - Push to Main
    ```git push origin main```


## Run the code
  
  - All code is currently launched through `python3 main.py`. Check hydra documentation (https://hydra.cc/) for overriding default parameters.
  - To run the debugging code: `python3 src/main.py +experiment=debug.yaml`. We advise to try to run the debug mode first
    before launching full experiments.
  - To run a code on only a few batches: `python3 main.py general.name=test`.
  - To run the continuous model: `python3 main.py model=continuous`
  - To run the discrete model: `python3 main.py`
  - You can specify the dataset with `python3 main.py dataset=guacamol`. Look at `configs/dataset` for the list
of datasets that are currently available
    
## Checkpoints

**My drive account has unfortunately been deleted, and I have lost access to the checkpoints. If you happen to have a downloaded checkpoint stored locally, I would be glad if you could send me an email at vignac.clement@gmail.com or raise a Github issue.**

The following checkpoints should work with the latest commit:

  - [QM9 (heavy atoms only)](https://drive.switch.ch/index.php/s/8IhyGE4giIW1wV3) \\
  
  - [Planar](https://drive.switch.ch/index.php/s/8IhyGE4giIW1wV3) \\

  - MOSES (the model in the paper was trained a bit longer than this one): https://drive.google.com/file/d/1LUVzdZQRwyZWWHJFKLsovG9jqkehcHYq/view?usp=sharing -- This checkpoint has been sent to me, but I have not tested it. \\

  - SBM: ~~https://drive.switch.ch/index.php/s/rxWFVQX4Cu4Vq5j~~ \\
    Performance of this checkpoint:
    - Test NLL: 4757.903
    - `{'spectre': 0.0060240439382095445, 'clustering': 0.05020166160905111, 'orbit': 0.04615866844490847, 'sbm_acc': 0.675, 'sampling/frac_unique': 1.0, 'sampling/frac_unique_non_iso': 1.0, 'sampling/frac_unic_non_iso_valid': 0.625, 'sampling/frac_non_iso': 1.0}`

  - Guacamol: https://drive.google.com/file/d/1KHNCnPJmPjIlmhnJh1RAvhmVBssKPqF4/view?usp=sharing -- This checkpoint has been sent to me, but I have not tested it.

## Generated samples

We provide the generated samples for some of the models. If you have retrained a model from scratch for which the samples are
not available yet, we would be very happy if you could send them to us!


## Troubleshooting 

`PermissionError: [Errno 13] Permission denied: '/home/vignac/DiGress/src/analysis/orca/orca'`: You probably did not compile orca.
    

## Use DiGress on a new dataset

To implement a new dataset, you will need to create a new file in the `src/datasets` folder. Depending on whether you are considering
molecules or abstract graphs, you can base this file on `moses_dataset.py` or `spectre_datasets.py`, for example. 
This file should implement a `Dataset` class to process the data (check [PyG documentation](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html)), 
as well as a `DatasetInfos` class that is used to define the noise model and some metrics.

For molecular datasets, you'll need to specify several things in the DatasetInfos:
  - The atom_encoder, which defines the one-hot encoding of the atom types in your dataset
  - The atom_decoder, which is simply the inverse mapping of the atom encoder
  - The atomic weight for each atom atype
  - The most common valency for each atom type

The node counts and the distribution of node types and edge types can be computed automatically using functions from `AbstractDataModule`.

Once the dataset file is written, the code in main.py can be adapted to handle the new dataset, and a new file can be added in `configs/dataset`.

### 1. Put your raw data in the project
Inside the repo root, create a data folder (if not already there):
```
DiGress/
├── data/
│   └── my_dataset/
│       ├── raw/
│       └── processed/   # will be auto-created
```
Put your original files (CSV, JSON, NPZ, SMILES, edge lists, etc.) inside: `data/my_dataset/raw/`

### 2. Create a dataset file
Create a new file, for example: 

`src/datasets/my_dataset.py`

### 3. Implement the Dataset class (PyG style)

Base it on:
- `moses_dataset.py` → for molecules
- `spectre_datasets.py` → for general graphs

Minimal Structure:
```
import os
import torch
from torch_geometric.data import InMemoryDataset, Data

class MyDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.csv']  # change to your file

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Not needed if data is local
        pass

    def process(self):
        data_list = []

        # ---- LOAD YOUR RAW DATA ----
        # Example: replace with your own logic
        import pandas as pd
        df = pd.read_csv(self.raw_paths[0])

        for i in range(len(df)):
            # Build graph
            edge_index = ...  # tensor shape [2, num_edges]
            x = ...           # node features
            edge_attr = ...   # optional

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr
            )

            data_list.append(data)

        # Optional transforms
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
```

### 4. Create DatasetInfos class
This is critical for DiGress (noise model + decoding).

Add in the same file:

```
class MyDatasetInfos:
    def __init__(self, datamodule):
        self.name = "my_dataset"

        # ---- REQUIRED ----
        self.num_node_types = datamodule.num_node_types
        self.num_edge_types = datamodule.num_edge_types

        # ---- FOR MOLECULES ONLY ----
        self.atom_encoder = {'C': 0, 'O': 1, 'N': 2}
        self.atom_decoder = {v: k for k, v in self.atom_encoder.items()}

        self.atomic_weights = {
            0: 12.0,  # C
            1: 16.0,  # O
            2: 14.0   # N
        }

        self.valencies = {
            0: 4,
            1: 2,
            2: 3
        }
```
If your dataset is not molecular, you can skip:
```
atom_encoder
atomic_weights
valencies
```
and just define:
```
self.num_node_types
self.num_edge_types
```

### 5. Compute dataset statistics automatically
DiGress already provides helpers via `AbstractDataModule`.

So you don’t manually compute:

- node count distribution
- edge type distribution

Just ensure your Data objects have:
```
x → node features (one-hot or categorical)
edge_attr → edge types (if applicable)
```

### 6. Register your dataset in the DataModule
Find where datasets are loaded (usually something like):
```
src/datasets/abstract_dataset.py
or
src/data/
```

Add your dataset:

```
from src.datasets.my_dataset import MyDataset, MyDatasetInfos
```

Then extend the logic:
```
if cfg.dataset.name == "my_dataset":
    dataset = MyDataset(root='data/my_dataset')
    dataset_infos = MyDatasetInfos(self)
```

### 7. Add a config file

Create:
```
configs/dataset/my_dataset.yaml
```
Example:
```
name: my_dataset

datadir: data/my_dataset

batch_size: 32
num_workers: 4

# graph sizes
max_n_nodes: 50

# feature sizes (important!)
num_node_types: 5
num_edge_types: 3
```

### 8. Modify `main.py`

Locate where dataset config is parsed.

Add your dataset option if needed:
```
if cfg.dataset.name == "my_dataset":
    ...
```

### 9. Run preprocessing

The first run will trigger PyG processing:

```python main.py dataset=my_dataset```

You should see:

```
Processing...
Done!
```


## Cite the paper

```
@inproceedings{
vignac2023digress,
title={DiGress: Discrete Denoising diffusion for graph generation},
author={Clement Vignac and Igor Krawczuk and Antoine Siraudin and Bohan Wang and Volkan Cevher and Pascal Frossard},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=UaAD-Nu86WX}
}
```
