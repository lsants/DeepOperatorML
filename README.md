# High Performance Numeric Integration

### Description

This project aims to develop a fast and high-performance method for integrating influence functions. The set of functions explored by this method frequently appear in various physical and engineering problems. Problems of this nature with unbounded domains are usually solved with the Boundary Element Method (BEM), which requires computing influence functions through integration at a significant number of points.

Traditional numerical integration techniques often struggle with these functions due to their complex nature, including singularities and improper integrals extending to infinity. By leveraging the field of operator learning and utilizing Deep Operator Networks (DeepONets), this project seeks to learn the underlying mathematical operators governing these problems, providing a more efficient and accurate integration approach that should reduce the overall cost of methods such as BEM.

### Project Overview

- **Goal**: To create a high-performance integration method for complex functions common in physical and engineering contexts, particularly those involving improper integrals with singularities and infinite upper limits.

- **Approach**: Utilize operator learning through DeepONets to model and learn the underlying mathematical operators that define the problem, enabling faster and more efficient computation.

- **Problem Statement**: Compute influence functions that describe the response of an isotropic half-space to a line load applied at a plane at the origin. These influence functions are computationally expensive to calculate due to the nature of the integrals involved.
  
- **Implementation**: The model is developed in Python using PyTorch's deep learning API.

---

## Project Structure

The repository is organized as follows:
```
📦 
.gitignore
README.md
comparison_plot.py
data_generation
│  ├─ axsgrsce.so
│  └─ influence.py
├─ data_generation_params.yaml
├─ get_data.py
├─ modules
deeponet_architecture.py
deeponet_inference.py
evaluate_params.py
│  ├─ mlp_architecture.py
│  ├─ mlp_training.py
│  ├─ plotting.py
│  └─ preprocessing.py
├─ params_model.yaml
├─ requirements.txt
├─ test_model.py
└─ train_model.py
```
