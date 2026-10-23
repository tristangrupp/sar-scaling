"""Colored outlines that mark what each image in a results panel is.

Every experiment's panel figures use the same two colors, so a reader can tell at a glance which tile
is the model's output and which is the truth it is compared against:

    green  = model output (any learned model's prediction)
    purple = comparison (the truth or reference the model is scored against)

Inputs and baselines (bicubic, Lee, the low-resolution input) get no outline.

    import sys; sys.path.insert(0, str(PROJ))      # project root
    from panel_roles import outline, MODEL, TRUTH
    outline(ax, "model"); outline(ax, "truth")
"""
from matplotlib.patches import Rectangle

MODEL = "#1f9d55"   # green
TRUTH = "#7e3fc0"   # purple
_COLORS = {"model": MODEL, "truth": TRUTH}


def outline(ax, role, lw=4):
    """Draw a colored frame just inside the axes edge. Works with ax.axis('off')."""
    color = _COLORS[role]
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor=color,
                           linewidth=lw, clip_on=False, zorder=10))
    ax.title.set_color(color)
    ax.title.set_fontweight("bold")


def legend(fig, y=0.005):
    """One line under the figure saying what the colors mean."""
    fig.text(0.5, y, "green outline = model output     purple outline = truth it is compared against",
             ha="center", va="bottom", fontsize=9, color="#444444")
