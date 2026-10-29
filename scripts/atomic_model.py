import abtem
import ase.io
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ase.data import chemical_symbols
from ase.data.colors import jmol_colors
from matplotlib.lines import Line2D

cif_path = Path(__file__).resolve().parents[1] / "atomic_model" / "SrTiO3.cif"
out_dir = Path(__file__).resolve().parent.parent / "figures"
out_dir.mkdir(exist_ok=True)

srtio3 = ase.io.read(cif_path)
srtio3 = srtio3.repeat((8, 8, 50))

# 物种图例（放到图形右侧之外，避免遮挡原子）
handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markeredgecolor="k",
        label=chemical_symbols[number],
        markerfacecolor=jmol_colors[number],
        markersize=12,
    )
    for number in np.unique(srtio3.numbers)
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
abtem.show_atoms(srtio3, ax=ax1, title="Beam view", scale=0.5, show_periodic=True)
abtem.show_atoms(srtio3, ax=ax2, plane="xz", title="Side view", scale=0.5, show_periodic=True)
fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.7, 0.5), frameon=False)
fig.subplots_adjust(right=0.8)
fig.savefig(out_dir / "atoms.png", dpi=300, bbox_inches="tight")
print(f"saved: {out_dir / 'atoms.png'}")

#plt.show()
