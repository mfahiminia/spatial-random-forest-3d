# -*- coding: utf-8 -*-
r"""
Methodology schematic: where spatial information enters a tree ensemble.

The argument of the paper is that the three spatially aware methods do not
differ in how much spatial information they use, but in WHERE it is injected
into an otherwise identical pipeline:

    RF + XYZ    -> the feature space   (coordinates appended as predictors)
    SRF_3D      -> the representation  (the sample becomes a k^3 pattern)
    GLS-RF_3D   -> the estimator       (split criterion and node representative
                                        become generalised least squares under
                                        an oriented covariance)

Ordinary RF leaves the pipeline unmodified. Drawing the three interventions
against a single shared pipeline makes the taxonomy visible in one image, and
explains why their signatures in the deployed model differ.

Run:  python -u methodology_figure.py
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle

try:
    _HERE = Path(__file__).resolve().parent
except NameError:                       # notebook cell
    _HERE = Path.cwd()
RF = Path(os.environ.get("SRF_PROJECT_ROOT", _HERE))
OUT = RF / "BlockModel" / "Coherence"

C_PIPE = "#5A5A5A"
C_PIPE_FILL = "#F2F2EF"
C_XYZ = "#56B4E9"
C_SRF = "#009E73"
C_GLS = "#0072B2"
C_RF = "#E69F00"

FS_STAGE = 9.0
FS_TITLE = 10.5
FS_BODY = 8.3
FS_SMALL = 7.4


def box(ax, x, y, w, h, fc, ec, lw=1.4, r=0.018, z=2, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
                       alpha=alpha)
    ax.add_patch(p)
    return p


def arrow(ax, xy_from, xy_to, color, lw=1.6, style="-|>", ls="-", z=4,
          rad=0.0):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle=style,
                                 mutation_scale=13, linewidth=lw, color=color,
                                 zorder=z, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


def draw_partition(ax, x0, y0, w, h, color):
    """Axis-aligned partition of space: what coordinate splits produce."""
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="white",
                           edgecolor=color, lw=1.1, zorder=3))
    for fx in (0.38, 0.72):
        ax.plot([x0 + fx * w, x0 + fx * w], [y0, y0 + h], color=color,
                lw=0.9, zorder=4)
    for fy, xa, xb in ((0.55, 0.0, 0.38), (0.33, 0.38, 1.0), (0.74, 0.72, 1.0)):
        ax.plot([x0 + xa * w, x0 + xb * w], [y0 + fy * h, y0 + fy * h],
                color=color, lw=0.9, zorder=4)


def draw_voxels(ax, cx, cy, s, color):
    """Isometric 3 x 3 x 3 neighbourhood."""
    dx, dy = 0.35 * s, 0.22 * s
    for layer, alpha in zip((2, 1, 0), (0.30, 0.55, 1.0)):
        ox, oy = cx - layer * dx * 0.5, cy - s * 0.5 + layer * dy
        for i in range(3):
            for j in range(3):
                ax.add_patch(Rectangle((ox + i * s / 3.0, oy + j * s / 3.0),
                                       s / 3.0 * 0.92, s / 3.0 * 0.92,
                                       facecolor=color, alpha=alpha * 0.45,
                                       edgecolor=color, lw=0.7, zorder=3))


def draw_ellipsoid(ax, cx, cy, color):
    """Oriented correlation ellipse with its principal axes."""
    ax.add_patch(Ellipse((cx, cy), 0.115, 0.048, angle=27, facecolor=color,
                         alpha=0.20, edgecolor=color, lw=1.2, zorder=3))
    ax.add_patch(Ellipse((cx, cy), 0.072, 0.030, angle=27, facecolor="none",
                         edgecolor=color, lw=0.8, ls=(0, (2, 2)), zorder=4))
    ax.annotate("", xy=(cx + 0.051, cy + 0.026), xytext=(cx - 0.051, cy - 0.026),
                arrowprops=dict(arrowstyle="-", color=color, lw=1.0), zorder=5)
    ax.annotate("", xy=(cx - 0.012, cy + 0.023), xytext=(cx + 0.012, cy - 0.023),
                arrowprops=dict(arrowstyle="-", color=color, lw=1.0), zorder=5)


def main():
    fig, ax = plt.subplots(figsize=(12.4, 6.9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---------------------------------------------------------------- pipeline
    stages = [
        ("Sampled node", "location $s_i$\ncovariates $z_i$"),
        ("Feature vector", "$x_i$ presented\nto the forest"),
        ("Split search", "score every\ncandidate split"),
        ("Node representative", "value assigned\nto the leaf"),
        ("Prediction", "$\\hat{y}(s_0)$ at an\nunsampled node"),
    ]
    w, gap, y0, h = 0.165, 0.025, 0.400, 0.135
    xs = [0.04 + k * (w + gap) for k in range(5)]
    centres = []
    for (x, (title, sub)) in zip(xs, stages):
        box(ax, x, y0, w, h, C_PIPE_FILL, C_PIPE, lw=1.3)
        ax.text(x + w / 2, y0 + h - 0.038, title, ha="center", va="center",
                fontsize=FS_STAGE, fontweight="bold", color="#222", zorder=5)
        ax.text(x + w / 2, y0 + 0.040, sub, ha="center", va="center",
                fontsize=FS_SMALL, color="#444", zorder=5)
        centres.append(x + w / 2)
    for k in range(4):
        arrow(ax, (xs[k] + w, y0 + h / 2), (xs[k + 1], y0 + h / 2), C_PIPE,
              lw=1.5)

    # ------------------------------------------------- upper interventions ---
    ytop, htop = 0.655, 0.295

    # RF + XYZ
    bx, bw = 0.045, 0.42
    box(ax, bx, ytop, bw, htop, "white", C_XYZ, lw=1.6)
    ax.text(bx + 0.016, ytop + htop - 0.035, "Route 1 — feature space",
            ha="left", va="center", fontsize=FS_BODY, color=C_XYZ,
            fontweight="bold")
    ax.text(bx + 0.016, ytop + htop - 0.072, "RF + XYZ", ha="left",
            va="center", fontsize=FS_TITLE, fontweight="bold", color="#222")
    ax.text(bx + 0.016, ytop + 0.072,
            "The coordinates are appended to the covariates,\n"
            "$x_i=[\\,z_i\\;|\\;x,y,z\\,]$, and are treated as ordinary\n"
            "predictors. Location can only be used through\n"
            "axis-aligned splits, which partition space into\n"
            "rectangular cells rather than along structure.",
            ha="left", va="center", fontsize=FS_SMALL, color="#333",
            linespacing=1.55)
    draw_partition(ax, bx + 0.318, ytop + 0.055, 0.088, 0.125, C_XYZ)

    # SRF_3D
    bx2, bw2 = 0.495, 0.46
    box(ax, bx2, ytop, bw2, htop, "white", C_SRF, lw=1.6)
    ax.text(bx2 + 0.016, ytop + htop - 0.035, "Route 2 — representation",
            ha="left", va="center", fontsize=FS_BODY, color=C_SRF,
            fontweight="bold")
    ax.text(bx2 + 0.016, ytop + htop - 0.072, "SRF_3D", ha="left",
            va="center", fontsize=FS_TITLE, fontweight="bold", color="#222")
    ax.text(bx2 + 0.016, ytop + 0.072,
            "The sample is replaced by its $3\\times3\\times3$ neighbourhood:\n"
            "27 voxels $\\times$ 5 channels = 135 features, plus 32\n"
            "anisotropy descriptors ($D=167$). Spatial context\n"
            "enters as pattern; no coordinate is ever used, so\n"
            "the model is translation-invariant by construction.",
            ha="left", va="center", fontsize=FS_SMALL, color="#333",
            linespacing=1.55)
    draw_voxels(ax, bx2 + 0.328, ytop + 0.135, 0.090, C_SRF)

    for src in (bx + bw * 0.62, bx2 + bw2 * 0.36):
        arrow(ax, (src, ytop), (centres[1], y0 + h),
              C_XYZ if src < 0.47 else C_SRF, lw=1.7, rad=0.0)

    # ------------------------------------------------- lower intervention ----
    ybot, hbot = 0.030, 0.290
    bx3, bw3 = 0.300, 0.655

    # Ordinary RF occupies the remaining corner, as the reference case.
    box(ax, 0.045, ybot, 0.225, hbot, "#FDF6E8", C_RF, lw=1.6)
    ax.text(0.061, ybot + hbot - 0.035, "Reference", ha="left", va="center",
            fontsize=FS_BODY, color=C_RF, fontweight="bold")
    ax.text(0.061, ybot + hbot - 0.072, "Ordinary RF", ha="left", va="center",
            fontsize=FS_TITLE, fontweight="bold", color="#222")
    ax.text(0.061, ybot + 0.085,
            "The pipeline above, unmodified.\n"
            "No spatial term enters at any\n"
            "stage, so any spatial mechanism\n"
            "must justify itself against this\n"
            "baseline.",
            ha="left", va="center", fontsize=FS_SMALL, color="#333",
            linespacing=1.55)
    arrow(ax, (0.1575, ybot + hbot), (0.1575, y0), C_RF, lw=1.5,
          style="-|>", ls=(0, (4, 2)))
    box(ax, bx3, ybot, bw3, hbot, "white", C_GLS, lw=1.6)
    ax.text(bx3 + 0.016, ybot + hbot - 0.035, "Route 3 — estimator",
            ha="left", va="center", fontsize=FS_BODY, color=C_GLS,
            fontweight="bold")
    ax.text(bx3 + 0.016, ybot + hbot - 0.072, "GLS-RF_3D", ha="left",
            va="center", fontsize=FS_TITLE, fontweight="bold", color="#222")
    ax.text(bx3 + 0.185, ybot + 0.095,
            "The feature vector is untouched — only the four covariates enter. "
            "Instead the\n"
            "objective is whitened by an oriented covariance "
            "$\\Sigma$ read from the experimental\n"
            "variogram (ranges 120 / 50 / 25 m; azimuth 110°, dip 25°, tilt 20°; "
            "nugget $g=0.20$).\n"
            "Both the split score and the leaf value become generalised least "
            "squares under\n"
            "$Q=\\Sigma^{-1}$:   "
            "$\\hat{\\beta}=(Z^{\\top}QZ)^{-1}Z^{\\top}QY$.   Nearby samples are "
            "thereby treated as\n"
            "partially redundant rather than as independent evidence.",
            ha="left", va="center", fontsize=FS_SMALL, color="#333",
            linespacing=1.62)
    draw_ellipsoid(ax, bx3 + 0.093, ybot + 0.120, C_GLS)
    ax.text(bx3 + 0.093, ybot + 0.058, "anisotropic $\\Sigma$", ha="center",
            va="center", fontsize=FS_SMALL, color=C_GLS, style="italic")

    for tgt in (centres[2], centres[3]):
        arrow(ax, (tgt - 0.02 + 0.04 * (tgt > 0.6), ybot + hbot),
              (tgt, y0), C_GLS, lw=1.7)

    fig.suptitle("Where spatial information enters the ensemble",
                 fontsize=13.5, x=0.012, ha="left", y=0.988,
                 fontweight="bold")
    fig.text(0.012, 0.948,
             "The three spatially aware methods modify different stages of one "
             "common pipeline; ordinary RF modifies none of them.",
             ha="left", fontsize=9.2, color="#555")

    fig.tight_layout(rect=[0, 0, 1, 0.945])
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_methodology.{ext}", dpi=300,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT/'fig_methodology.png'} and .pdf")


if __name__ == "__main__":
    main()
