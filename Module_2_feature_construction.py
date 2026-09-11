"""
Module 2 : Module_2_feature_construction.py
"""

import numpy as np
from itertools import combinations_with_replacement
from Module_1_lorenz63 import apply_style


class NGRCFeatureConstructor:

    def __init__(self, delay=2, skip=1):
        self.k = delay
        self.s = skip

    def load_dataset(self, filename="Data/lorenz63_dataset.npy"):
        data = np.load(filename)
        print("Dataset Loaded. Shape :", data.shape)
        return data

    def delay_embedding(self, data):
        X = data[:, 1:]  # drop time column
        start = self.k * self.s

        linear = np.array([
            np.concatenate([X[i - j * self.s] for j in range(self.k)])
            for i in range(start, len(X))
        ])

        print("Linear Feature Shape :", linear.shape)
        return linear

    def quadratic_features(self, linear):
        n = linear.shape[1]
        indices = list(combinations_with_replacement(range(n), 2))

        quadratic = np.stack(
            [linear[:, i] * linear[:, j] for i, j in indices],
            axis=1
        )

        print("Quadratic Feature Shape :", quadratic.shape)
        return quadratic, indices

    def total_features(self, linear, quadratic):
        reservoir = np.hstack((
            np.ones((linear.shape[0], 1)),
            linear,
            quadratic
        ))

        print("Total Reservoir Shape :", reservoir.shape)
        return reservoir

    def feature_names(self, quad_indices):
        variables = [
            "x(t)", "y(t)", "z(t)",
            "x(t-1)", "y(t-1)", "z(t-1)"
        ]

        names = ["const"] + variables

        names += [
            f"{variables[i]}²" if i == j
            else f"{variables[i]}·{variables[j]}"
            for i, j in quad_indices
        ]

        return names

    def save(self, reservoir, names):
        np.save("Data/ngrc_reservoir.npy", reservoir)
        np.save("Data/feature_names.npy", np.array(names))

        print("Reservoir Saved: Data/ngrc_reservoir.npy")
        print("Feature Names Saved: Data/feature_names.npy")


if __name__ == "__main__":

    apply_style()

    fc = NGRCFeatureConstructor(delay=2, skip=1)

    dataset = fc.load_dataset()
    linear = fc.delay_embedding(dataset)
    quadratic, quad_idx = fc.quadratic_features(linear)

    reservoir = fc.total_features(linear, quadratic)
    names = fc.feature_names(quad_idx)

    fc.save(reservoir, names)

    print("\nModule 2 Completed Successfully.")