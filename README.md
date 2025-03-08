# DeepONet For Soil-Structure Interaction

### Description

This project aims to develop a framework for solving soil-structure interaction problems using Deep Learning. Problems of this nature usually have unbounded domains and are usually solved with the Boundary Element Method (BEM), which requires computing influence functions through costly numerical integration schemes at a significant number of points.

Traditional numerical integration techniques often struggle with such functions due to their complex nature, including singularities and improper integrals extending to infinity. By leveraging the field of operator learning and utilizing Deep Operator Networks (DeepONets), this project seeks to bypass the use of BEM and solve the differential equations directly by using data.

### Project Overview

- **Goal**: To create a PDE solver that is capable of efficiently solving soil-structure interaction problems.

- **Approach**: Employment of DeepONets with multiple training strategies.

- **Problems**: The following problems have been implemented so far:
  - Kelvin's problem (static response of an isotropic elastic 3D space to a point load).
  - Homogeneous Green functions (harmonic response of an isotropic elastic 3D space to a point load).
  
- **Implementation**: The model is developed in Python using PyTorch.

---

## Repository Structure

The repository is organized as follows:
```
📦 
├─ .gitignore
├─ README.md
├─ configs
│  ├─ config_data_generation.yaml
│  ├─ config_test.yaml
│  └─ config_train.yaml
├─ get_data.py
├─ main.py
├─ requirements.txt
├─ run_experiments.py
└─ src
   ├─ __init__.py
   ├─ modules
   │  ├─ __init__.py
   │  ├─ data_generation
   │  │  ├─ __init__.py
   │  │  ├─ axsgrsce.dll
   │  │  ├─ axsgrsce.dylib
   │  │  ├─ axsgrsce.so
   │  │  ├─ data_generation_base.py
   │  │  ├─ data_generation_dynamic_fixed_material.py
   │  │  ├─ data_generation_kelvin.py
   │  │  └─ influence.py
   │  ├─ data_processing
   │  │  ├─ __init__.py
   │  │  ├─ compose_transformations.py
   │  │  └─ deeponet_dataset.py
   │  ├─ deeponet
   │  │  ├─ __init__.py
   │  │  ├─ components
   │  │  │  ├─ __init__.py
   │  │  │  ├─ base_branch.py
   │  │  │  ├─ base_trunk.py
   │  │  │  ├─ pod_trunk.py
   │  │  │  ├─ pre_trained_trunk.py
   │  │  │  ├─ trainable_branch.py
   │  │  │  └─ trainable_trunk.py
   │  │  ├─ deeponet.py
   │  │  ├─ factories
   │  │  │  ├─ __init__.py
   │  │  │  ├─ activation_factory.py
   │  │  │  ├─ component_factory.py
   │  │  │  ├─ loss_factory.py
   │  │  │  ├─ model_factory.py
   │  │  │  ├─ network_factory.py
   │  │  │  ├─ optimizer_factory.py
   │  │  │  └─ strategy_factory.py
   │  │  ├─ nn
   │  │  │  ├─ __init__.py
   │  │  │  ├─ activation_fns.py
   │  │  │  ├─ kan.py
   │  │  │  ├─ mlp.py
   │  │  │  ├─ net.py
   │  │  │  ├─ network_architectures.py
   │  │  │  └─ resnet.py
   │  │  ├─ optimization
   │  │  │  ├─ __init__.py
   │  │  │  ├─ loss_fns.py
   │  │  │  └─ optimizers.py
   │  │  ├─ output_handling
   │  │  │  ├─ __init__.py
   │  │  │  ├─ output_handling_base.py
   │  │  │  ├─ share_branch.py
   │  │  │  ├─ share_trunk.py
   │  │  │  ├─ single_output.py
   │  │  │  └─ split_outputs.py
   │  │  └─ training_strategies
   │  │     ├─ __init__.py
   │  │     ├─ helpers
   │  │     │  ├─ __init__.py
   │  │     │  ├─ decomposition_helper.py
   │  │     │  ├─ phase_manager.py
   │  │     │  ├─ pod_helper.py
   │  │     │  └─ two_step_helper.py
   │  │     ├─ pod_training.py
   │  │     ├─ standard_training.py
   │  │     ├─ training_strategy_base.py
   │  │     └─ two_step_training.py
   │  ├─ pipe
   │  │  ├─ __init__.py
   │  │  ├─ inference.py
   │  │  ├─ optimizer_manager.py
   │  │  ├─ preprocessing.py
   │  │  ├─ saving.py
   │  │  ├─ store_ouptuts.py
   │  │  └─ training.py
   │  ├─ plotting
   │  │  ├─ __init__.py
   │  │  ├─ plot_axis.py
   │  │  ├─ plot_basis.py
   │  │  ├─ plot_field.py
   │  │  └─ plot_training.py
   │  └─ utilities
   │     ├─ __init__.py
   │     ├─ config_utils.py
   │     ├─ dir_functions.py
   │     └─ log_functions.py
   ├─ test.py
   └─ train.py
```
©generated by [Project Tree Generator](https://woochanleee.github.io/project-tree-generator)
## Data generation

The data for training the DeepOnet can be generated by defining the boundary value problem's parameters in the  ```/configs/config_data_generation.yaml``` file and running the ```get_data.py``` script with the ```--problem``` flag with the desired problem.

## DeepONet trainning

To train or test a model, define the model and training/testing parameters in the ```/configs/config_train.yaml```/```/configs/config_test.yaml``` file and run ```main.py```.
