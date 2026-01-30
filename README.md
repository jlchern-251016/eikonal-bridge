# The Eikonal Bridge

**From Classical Lens Design to Quantum Photonics via Differentiable Computing**

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-enabled-green.svg)](https://github.com/google/jax)
[![SPIE Press](https://img.shields.io/badge/Publisher-SPIE%20Press-orange.svg)](https://spie.org)

---

## Overview

This repository contains the companion materials for *The Eikonal Bridge*, a comprehensive technical book that unifies classical optical design with quantum photonics through the power of differentiable computing.

**The Central Innovation**: The Differentiable Eikonal Engine (DEE) framework uses JAX automatic differentiation to bridge two traditionally separate worlds—classical lens design and quantum photonics—through a single elegant identity:

```
φ_quantum = 2π × W_eikonal / λ
```

This **bridge identity** reveals that the classical eikonal function (optical path length) and quantum optical phase are the same mathematical object, enabling unified computational tools for both domains.

---

## Who This Book Is For

- **Optical Engineers** transitioning from CODE V/Zemax to quantum photonics applications
- **Quantum Physicists** seeking practical lens design foundations for photonic systems
- **Graduate Students** in optics, photonics, and quantum information science
- **Researchers** exploring differentiable programming for optical design

---

## The W/MN Duality Framework

Every chapter presents optical problems from dual perspectives:

| Perspective | Question | Approach |
|-------------|----------|----------|
| **Walther (Forward)** | "Given this device, what does it do?" | Analysis, simulation, characterization |
| **Matsui-Nariai (Inverse)** | "Given this target, how do I build it?" | Design, optimization, synthesis |

This duality provides a universal problem-solving methodology that applies across classical and quantum domains.

---

## Three-Axis Failure Framework

When does the scalar eikonal approximation break down? Our systematic diagnostic framework identifies failure modes across three axes:

| Axis | Codes | Failure Modes |
|------|-------|---------------|
| **Physical** | P1-P5 | High NA, polarization, dispersion, nonlinearity, thermal |
| **Mathematical** | M1-M5 | Singularities, discontinuities, non-differentiable, ill-conditioned, numerical precision |
| **Topological** | T1-T3 | OAM, Berry phase, topological charge |

---

## Book Structure

### Part I: Classical Foundations (Chapters 1-3)
- Ch 1: Introduction & Foundations
- Ch 2: Classical Eikonal Theory
- Ch 3: Aberration Theory via Eikonal

### Part II: Differentiable Computing (Chapters 4-6)
- Ch 4: Differentiable Computing Primer
- Ch 5: The Differentiable Eikonal Engine
- Ch 6: Classical Lens Design with DEE

### Part III: The Quantum Bridge (Chapters 7-10)
- Ch 7: The Classical-Quantum Bridge
- Ch 8: Quantum State Engineering
- Ch 9: Quantum Sensing Applications
- Ch 10: Quantum Imaging Systems

### Part IV: Advanced Topics & Future (Chapters 11-12)
- Ch 11: Advanced Topics (Vector Eikonal, Coherence, Topology)
- Ch 12: Future Directions

### Appendices
- A: Mathematical Foundations
- B: JAX Reference Guide
- C: CODE V/Zemax to DEE Translation
- D: Failure Mode Catalog

---

## Repository Structure

```
the-eikonal-bridge/
├── manuscript/           # LaTeX source files
│   ├── chapters/         # Individual chapter .tex files
│   ├── appendices/       # Appendix .tex files
│   ├── figures/          # Publication figures
│   └── main.tex          # Master document
├── code/                 # Python/JAX implementations
│   ├── dee_core/         # DEE framework core modules
│   ├── examples/         # Worked examples from each chapter
│   ├── problems/         # Problem solution code
│   └── figures/          # Figure generation scripts
├── solutions/            # Complete problem solutions manual
├── notebooks/            # Jupyter notebooks for exploration
├── data/                 # Sample datasets and results
└── docs/                 # Additional documentation
```

---

## Quick Start

### Installation

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

### Dependencies

```
numpy>=1.21.0
matplotlib>=3.5.0
jax>=0.4.0
jaxlib>=0.4.0
scipy>=1.7.0
pyyaml>=6.0
```

### Your First DEE Calculation

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

## Specification Translation Tables

A key feature of this book is quantitative guidance on tolerance tightening when transitioning from classical to quantum applications:

| Parameter | Classical Spec | Quantum Spec | Tightening Factor |
|-----------|---------------|--------------|-------------------|
| Wavefront Error | λ/14 RMS | λ/63 RMS | ~4.5× |
| Surface Figure | λ/20 P-V | λ/100 P-V | 5× |
| Temperature Stability | ±2 K | ±0.1 K | 20× |
| Alignment (angular) | 10 μrad | 0.1 μrad | 100× |
| Vibration Isolation | 10 nm RMS | 0.1 nm RMS | 100× |

---

## Key Features

✅ **Unified Framework**: Single computational infrastructure for classical and quantum optics  
✅ **Production-Ready Code**: All JAX implementations tested and documented  
✅ **W/MN Duality**: Every topic from both analysis and design perspectives  
✅ **Failure Diagnostics**: Systematic framework for identifying scalar eikonal limits  
✅ **Practical Focus**: Pain-point-first pedagogy addressing real engineering challenges  
✅ **Complete Solutions**: Full solution manual for all chapter problems  

---

## Citation

If you use this work in your research, please cite:

```bibtex
@book{lin2026eikonal,
  title     = {The Eikonal Bridge: From Classical Lens Design to 
               Quantum Photonics via Differentiable Computing},
  author    = {Chern, Jyh-Long},
  year      = {2026}
}
```

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

- **Bug Reports**: Open an issue with a minimal reproducible example
- **Feature Requests**: Describe the use case and proposed solution
- **Code Contributions**: Fork, create a feature branch, submit a PR

---

## License

- **Book Content**: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- **Code**: [MIT License](LICENSE-CODE)

---

## Acknowledgments

This work bridges decades of optical engineering wisdom with modern computational methods. Special thanks to the optical design community and the JAX development team.

---

## Contact

- **Author**: Jyh-Long Chern
- **Email**: []
- **LinkedIn**: [https://www.linkedin.com/in/jyh-long-chern-236a95141]

---

*"The eikonal function is not just a mathematical convenience—it is the Rosetta Stone connecting classical ray optics to quantum wave mechanics."*
