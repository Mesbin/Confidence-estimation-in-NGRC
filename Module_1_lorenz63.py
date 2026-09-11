"""

Module 1 : Module_1_lorenz63.py

"""

import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# --------------------------------------------------------------------------
# Publication figure style, imported by every module.
# 400 dpi, base font 15 pt, line width 2.8, no grid lines, no box.
# --------------------------------------------------------------------------
def apply_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 400,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "font.family": "DejaVu Sans", "font.size": 15,
        "axes.titlesize": 16, "axes.labelsize": 15,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
        "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.6, "axes.labelpad": 8, "axes.titlepad": 12,
        "lines.linewidth": 2.8, "lines.markersize": 8, "lines.markeredgewidth": 0,
        "patch.linewidth": 0,
        "xtick.major.width": 1.6, "ytick.major.width": 1.6,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "errorbar.capsize": 4,
    })


def save_figure(filename):
    """Save at 400 dpi, display, report."""
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()
    print(f"Saved {filename}")



class Lorenz63:

    def __init__(self, sigma=10.0, rho=28.0, beta=8/3, dt=0.01, total_time=100):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.total_time = total_time

    def equations(self, t, state):
        x, y, z = state
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return [dx, dy, dz]

    def generate(self, initial_state=[1.0, 1.0, 1.0]):
        t_eval = np.arange(0, self.total_time, self.dt)
        solution = solve_ivp(
            self.equations, (0, self.total_time), initial_state,
            t_eval=t_eval, method="RK45", rtol=1e-9, atol=1e-9
        )
        self.time, self.x, self.y, self.z = solution.t, *solution.y
        return self.time, self.x, self.y, self.z

    def dataset(self):
        return np.column_stack((self.time, self.x, self.y, self.z))

    def plot_attractor(self, filename="Results/lorenz_attractor.png"):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection='3d')
        ax.plot(self.x, self.y, self.z, linewidth=0.4, color="teal")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title("Lorenz-63 Attractor")
        plt.savefig(filename)
        plt.show()
        plt.close()
        print(f"Saved {filename}")

    def save(self, filename="Data/lorenz63_dataset.npy"):
        np.save(filename, self.dataset())
        print(f"Dataset saved as {filename}")


if __name__ == "__main__":
    apply_style()
    lorenz = Lorenz63()
    lorenz.generate()
    print("Dataset Shape:", lorenz.dataset().shape)
    lorenz.plot_attractor()
    lorenz.save()
    print("\nModule 1 Completed Successfully.")
