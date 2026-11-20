"""Render a small animated diagram of the project's wide-and-deep model."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "wide_and_deep_network.gif"

BG = "#0b1220"
TEXT = "#e5edf7"
MUTED = "#98a9bf"
WIDE = "#38bdf8"
DEEP = "#a78bfa"
JOIN = "#fbbf24"
PREDICTION = "#34d399"


def node_column(ax, x, center_y, count, color, label, detail):
    ys = np.linspace(center_y - 0.16, center_y + 0.16, count)
    ax.scatter([x] * count, ys, s=82, color=color, edgecolors=BG, linewidths=1.4, zorder=3)
    ax.text(x, center_y - 0.25, label, ha="center", va="top", color=TEXT,
            fontsize=10, fontweight="bold")
    ax.text(x, center_y - 0.32, detail, ha="center", va="top", color=MUTED,
            fontsize=8)
    return [(x, y) for y in ys]


def connect(ax, left, right, color, alpha=0.27):
    # Draw a representative, readable subset of the dense layer connections.
    for i, p0 in enumerate(left):
        for j, p1 in enumerate(right):
            if (i + j) % 2 == 0 or len(left) == 1 or len(right) == 1:
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color,
                        alpha=alpha, linewidth=0.8, zorder=1)


def main():
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0.02, 0.99)
    ax.set_ylim(0.05, 0.98)
    ax.axis("off")
    fig.subplots_adjust(left=0.025, right=0.985, top=0.86, bottom=0.12)

    ax.text(0.5, 0.96, "CALIFORNIA HOUSING  ·  WIDE & DEEP REGRESSOR",
            color=TEXT, ha="center", va="top", fontsize=16, fontweight="bold")
    ax.text(0.5, 0.89, "Blue: wide path  ·  Purple: deep path  ·  Gold: merge  ·  Green: prediction",
            color=MUTED, ha="center", va="top", fontsize=10)

    wide_input = node_column(ax, 0.11, 0.80, 5, WIDE, "", "")
    deep_input = node_column(ax, 0.11, 0.43, 5, DEEP, "Deep input", "16 features")
    hidden_64 = node_column(ax, 0.31, 0.43, 5, DEEP, "Dense + ReLU", "64 units")
    hidden_32 = node_column(ax, 0.49, 0.43, 5, DEEP, "Dense + ReLU", "32 units")
    merged = node_column(ax, 0.70, 0.59, 5, JOIN, "Concatenate", "14 + 32 values")
    output = node_column(ax, 0.91, 0.59, 1, PREDICTION, "Linear output", "predicted value")

    connect(ax, deep_input, hidden_64, DEEP)
    connect(ax, hidden_64, hidden_32, DEEP)
    connect(ax, hidden_32, merged, DEEP)
    connect(ax, wide_input, merged, WIDE)
    connect(ax, merged, output, PREDICTION, alpha=0.48)

    # A wide path bypasses the hidden layers; the deep path learns a nonlinear representation.
    wide_route = np.array([[0.16, 0.80], [0.38, 0.80], [0.55, 0.72], [0.68, 0.65]])
    deep_route = np.array([[0.16, 0.43], [0.27, 0.43], [0.45, 0.43], [0.57, 0.49], [0.68, 0.54]])
    out_route = np.array([[0.74, 0.59], [0.84, 0.59], [0.89, 0.59]])

    wide_dots = ax.scatter([], [], s=42, color=WIDE, edgecolors="white", linewidths=0.6, zorder=5)
    deep_dots = ax.scatter([], [], s=42, color=DEEP, edgecolors="white", linewidths=0.6, zorder=5)
    output_dots = ax.scatter([], [], s=48, color=PREDICTION, edgecolors="white", linewidths=0.6, zorder=5)
    phase_text = ax.text(0.5, 0.035, "", ha="center", va="center", color=TEXT, fontsize=9)

    def point_on_route(route, progress):
        lengths = np.linalg.norm(np.diff(route, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        distance = progress * cumulative[-1]
        segment = min(np.searchsorted(cumulative, distance, side="right") - 1, len(lengths) - 1)
        local = (distance - cumulative[segment]) / lengths[segment]
        return route[segment] * (1 - local) + route[segment + 1] * local

    def update(frame):
        progress = (frame % 48) / 47
        wide_positions = np.array([point_on_route(wide_route, (progress + offset) % 1) for offset in (0, 0.2, 0.4)])
        deep_positions = np.array([point_on_route(deep_route, (progress + offset) % 1) for offset in (0, 0.2, 0.4)])
        output_positions = np.array([point_on_route(out_route, (progress + offset) % 1) for offset in (0, 0.33, 0.66)])
        wide_dots.set_offsets(wide_positions)
        deep_dots.set_offsets(deep_positions)
        output_dots.set_offsets(output_positions)
        phase_text.set_text("Wide features pass directly  ·  Deep features learn  ·  Both paths join for prediction")
        return wide_dots, deep_dots, output_dots, phase_text

    animation = FuncAnimation(fig, update, frames=72, interval=60, blit=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    animation.save(OUTPUT, writer=PillowWriter(fps=15), dpi=100)
    plt.close(fig)
    print(f"Saved {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
