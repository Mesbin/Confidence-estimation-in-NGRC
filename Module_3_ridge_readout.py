"""
Module 3 : Module_3_ridge_readout.py

"""

import numpy as np
import matplotlib.pyplot as plt
from Module_1_lorenz63 import apply_style, save_figure


class RidgeReadout:

    def __init__(self, alpha=2.5e-6):
        self.alpha = alpha
        self.Wout = None

    def load_reservoir(self, filename="Data/ngrc_reservoir.npy"):
        reservoir = np.load(filename)
        print("Reservoir Loaded. Shape :", reservoir.shape)
        return reservoir

    def load_dataset(self, filename="Data/lorenz63_dataset.npy"):
        data = np.load(filename)
        print("Dataset Loaded. Shape :", data.shape)
        return data

    def create_target(self, data, k=2, s=1):
        start = k * s
        target = data[start + 1:, 1:]  # remove time column
        print("Target Shape :", target.shape)
        return target

    def align_reservoir(self, reservoir, target):
        reservoir = reservoir[:len(target)]
        print("Aligned Reservoir Shape :", reservoir.shape)
        return reservoir

    def train(self, reservoir, target):
        
        O = reservoir.T
        Y = target.T
        identity = np.identity(O.shape[0])
        self.Wout = (Y @ O.T) @ np.linalg.inv(O @ O.T + self.alpha * identity)
        print("Training Complete. Wout Shape :", self.Wout.shape)
        return self.Wout

    def save_weights(self, filename="Data/Wout.npy"):
        np.save(filename, self.Wout)
        print(f"Weights Saved to {filename}")

    # ==========================================================
if __name__ == "__main__":
    apply_style()
    model = RidgeReadout(alpha=2.5e-6)
    reservoir = model.load_reservoir()
    dataset = model.load_dataset()
    target = model.create_target(dataset)
    reservoir = model.align_reservoir(reservoir, target)
    model.train(reservoir, target)
    model.save_weights()
    print("\nModule 3 Completed Successfully.")