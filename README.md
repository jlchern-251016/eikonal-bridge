# The Eikonal Bridge 🌉

**From Classical Lens Design to Quantum Photonics via Differentiable Computing**

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-enabled-green.svg)](https://github.com/google/jax)
[![SPIE Press](https://img.shields.io/badge/Publisher-SPIE%20Press-orange.svg)](https://spie.org)
[![HTML5](https://img.shields.io/badge/Companion-Interactive%20Tools-red.svg)](https://www.w3.org/html/)

---

## 🎯 Overview


This repository contains the companion materials for *The Eikonal Bridge*, a comprehensive technical book that unifies classical optical design with quantum photonics through the power of differentiable computing.

**The Central Innovation**: The Differentiable Eikonal Engine (DEE) framework uses JAX automatic differentiation to bridge two traditionally separate worlds—classical lens design and quantum photonics—through a single elegant identity:

```
φ_quantum = 2π × W_eikonal / λ
```

This **bridge identity** reveals that the classical eikonal function (optical path length) and quantum optical phase are the same mathematical object, enabling unified computational tools for both domains.

---

## 🌟 Key Features

- **📚 12 Chapters in 4 Parts** - Foundations → Computational Optics → Quantum Extensions → Production Practice
- **🔬 W/MN Duality Framework** - Every topic from both forward analysis (Walther) and inverse design (Matsui-Nariai) perspectives
- **🎨 Production-Ready JAX Code** - Complete, tested implementations with 100-1000× speedup
- **⚠️ Three-Axis Failure Framework** - Systematic diagnostics for scalar eikonal limitations (P1-P5, M1-M5, T1-T3)
- **📊 Specification Translation Tables** - Quantitative classical→quantum tolerance tightening factors (10-100×)
- **🔧 CODE V/Zemax Migration Guide** - Appendix C bridges commercial tools to DEE
- **📝 80+ Paired Problems** - W-type (analysis), MN-type (design), and Q-type (quantum) problem sets

---

## 👥 Who This Book Is For

- **Optical Engineers** transitioning from CODE V/Zemax to quantum photonics applications
- **Quantum Physicists** seeking practical lens design foundations for photonic systems
- **Graduate Students** in optics, photonics, and quantum information science
- **Researchers** exploring differentiable programming for optical design

---

## 🔄 The W/MN Duality Framework

Every chapter presents optical problems from dual perspectives:

| Perspective | Question | Approach |
|-------------|----------|----------|
| **Walther (Forward)** 🔵 | "Given this device, what does it do?" | Analysis, simulation, characterization |
| **Matsui-Nariai (Inverse)** 🔴 | "Given this target, how do I build it?" | Design, optimization, synthesis |

This duality provides a universal problem-solving methodology that applies across classical and quantum domains.

---

## ⚠️ Three-Axis Failure Framework

When does the scalar eikonal approximation break down? Our systematic diagnostic framework identifies failure modes across three axes:

| Axis | Codes | Failure Modes |
|------|-------|---------------|
| **Physical** | P1-P5 | High NA, polarization, dispersion, nonlinearity, thermal |
| **Mathematical** | M1-M5 | Singularities, discontinuities, non-differentiable, ill-conditioned, numerical precision |
| **Topological** | T1-T3 | OAM, Berry phase, topological charge |

---

## 📖 Book Structure

### Part I: Foundations (Chapters 1–4)
| Ch | Title | Key Topics |
|----|-------|------------|
| 1 | Eikonal as Universal Language | Eikonal equation, bridge identity, singlet lens example |
| 2 | Hamilton's Characteristic Functions | V, T, W, W' functions; Seidel aberrations |
| 3 | Wavefront Aberrations | Zernike polynomials, PSF/MTF, Double Gauss example |
| 4 | Beyond Scalar Eikonal | Three-axis failure framework (P/M/T), Level 1–5 hierarchy |

### Part II: Computational Optics (Chapters 5–8)
| Ch | Title | Key Topics |
|----|-------|------------|
| 5 | Photonic Integration | Dimensional hierarchy, eigenmode-eikonal, CPO couplers |
| 6 | Quantum-Inspired Optimization | QUBO formulation, quantum annealing algorithms |
| 7 | Differentiable Eikonal Engine | DEE architecture, JAX implementation, benchmarks |
| 8 | Walther-(Matsui-Nariai) Duality | W/MN unified workflow, central methodology chapter |

### Part III: Quantum Extensions (Chapters 9–11)
| Ch | Title | Key Topics |
|----|-------|------------|
| 9 | Quantum Wavefront Sensing | Squeezed light sensing, MPLC mode sorter |
| 10 | Quantum Walks in Waveguide Arrays | Quantum walks, HOM effect, QRNG design |
| 11 | N-Photon Phase Multiplication | N-photon enhancement, NOON states, Heisenberg limit |

### Part IV: Practice (Chapter 12)
| Ch | Title | Key Topics |
|----|-------|------------|
| 12 | Production Workflows | AR/VR combiner, CPO coupler, metalens, QKD source |


---

## 📁 Repository Structure

```
the-eikonal-bridge/
├── assets/
│   └── images/               # README and documentation images
│       └── eikonal_evolution_bridge.png
├── manuscript/               # LaTeX source files
│   ├── chapters/             # Individual chapter .tex files
│   ├── appendices/           # Appendix .tex files
│   ├── figures/              # Publication figures
│   └── main.tex              # Master document
├── code/                     # Python/JAX implementations
│   ├── dee_core/             # DEE framework core modules
│   ├── examples/             # Worked examples from each chapter
│   ├── problems/             # Problem solution code
│   └── figures/              # Figure generation scripts
├── solutions/                # Complete problem solutions manual
├── notebooks/                # Jupyter notebooks for exploration
├── data/                     # Sample datasets and results
└── docs/                     # Additional documentation
```

---

## 🚀 Quick Start

### Online Usage (Recommended)
1. Clone or download the repository
2. Navigate to `notebooks/` for interactive exploration
3. No heavy installation required for basic exploration!

### Full Installation

```bash
# Clone the repository
git clone https://github.com/[username]/the-eikonal-bridge.git
cd the-eikonal-bridge

# Create virtual environment
python -m venv dee_env
source dee_env/bin/activate  # On Windows: dee_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 📦 Dependencies

```
numpy>=1.21.0
matplotlib>=3.5.0
jax>=0.4.0
jaxlib>=0.4.0
scipy>=1.7.0
pyyaml>=6.0
```

### 🎮 Your First DEE Calculation

```python
import jax.numpy as jnp
from jax import grad

def eikonal_to_quantum_phase(W_eikonal, wavelength):
    """
    Bridge identity: phi_quantum = 2*pi * W_eikonal / lambda
    """
    return 2 * jnp.pi * W_eikonal / wavelength

# The gradient is automatically available
d_phase_dW = grad(eikonal_to_quantum_phase)

# Example: 1 wave of OPD at 1.55 um
W = 1.55e-6  # meters
wl = 1.55e-6  # meters
phase = eikonal_to_quantum_phase(W, wl)
print(f"Quantum phase: {phase:.4f} rad = {phase/(2*jnp.pi):.4f} waves")
```

---

## 📊 Specification Translation Tables

A key feature of this book is quantitative guidance on tolerance tightening when transitioning from classical to quantum applications:

| Parameter | Classical Spec | Quantum Spec | Tightening Factor |
|-----------|---------------|--------------|-------------------|
| Wavefront Error | λ/14 RMS | λ/63 RMS | ~4.5× |
| Surface Figure | λ/20 P-V | λ/100 P-V | 5× |
| Temperature Stability | ±2 K | ±0.1 K | 20× |
| Alignment (angular) | 10 μrad | 0.1 μrad | 100× |
| Vibration Isolation | 10 nm RMS | 0.1 nm RMS | 100× |

---

## 🔬 Technical Highlights

### DEE Performance Benchmarks

| Operation | Finite Difference | JAX Autodiff | Speedup |
|-----------|-------------------|--------------|---------|
| Gradient (N=100) | 2.1 s | 0.003 s | 700× |
| Hessian (N=100) | 210 s | 0.15 s | 1400× |
| Optimization (1000 iter) | 35 min | 45 s | 47× |

### Eigenvalue-Eikonal Identity

The key equation connecting waveguide coupling to quantum state evolution:

```
β_k = (dW_k/dz) × (2π/λ)
```

This identity enables unified design of classical CPO couplers and quantum photonic gates.

---



## 🙏 Acknowledgments

This work bridges decades of optical engineering wisdom with modern computational methods. Special thanks to:
- The optical design community for foundational knowledge
- The JAX development team for enabling differentiable computing

---

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@book{lin2026eikonal,
  title     = {The Eikonal Bridge: From Classical Lens Design to 
               Quantum Photonics via Differentiable Computing},
  author    = {Lin, Jyh-Long},
  year      = {2025},
  publisher = {Open-source},
  series    = {},
  isbn      = {}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

- **🐛 Bug Reports**: Open an issue with a minimal reproducible example
- **💡 Feature Requests**: Describe the use case and proposed solution
- **🔧 Code Contributions**: Fork, create a feature branch, submit a PR

---

## 📜 License

- **Book Content**: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- **Code**: [MIT License](LICENSE-CODE)

---

## 🚦 Project Status

**Current Version**: 1.0 (Publication Preparation)  
**Last Updated**: January 2026  
**Active Development**: Yes


---


*"The eikonal function is not just a mathematical convenience—it is the Rosetta Stone connecting classical ray optics to quantum wave mechanics."* 

---

