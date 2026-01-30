# The Eikonal Bridge
**From classical lens design to quantum photonics through differentiable computing**

This repository is the open-source home for the book draft and companion materials:
- Book source (LaTeX) + figures  (Not released yet)
- Reproducible PDF builds (CI)
- Companion code (Python/JAX) and tool hooks (Code V / LightTools / RSoft) (Not released yet)
- Versioned releases (PDF artifacts)

## Quick links
- **Read online (GitHub Pages):** see the `docs/` site once enabled.
- **Download PDF:** go to **Releases** (tagged versions).
- **Build locally:** instructions below.

## Build locally (recommended)
### Option A — latexmk (TeX Live)
```bash
cd book/tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../out main.tex
