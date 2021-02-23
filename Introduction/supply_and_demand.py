import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


class Equilibrium:
    def __init__(self, α=0.1, β=1, γ=1, δ=1):
        self.α, self.β, self.γ, self.δ = α, β, γ, δ

    def qs(self, p):
        return np.exp(self.α * p) - self.β

    def qd(self, p):
        return self.γ * p ** (-self.δ)

    def compute_equilibrium(self):
        def h(p):
            return self.qd(p) - self.qs(p)

        p_star = brentq(h, 2, 4)
        q_star = np.exp(self.α * p_star) - self.β

        print(f"Equilibrium price is {p_star: .2f}")
        print(f"Equilibrium quantity is {q_star: .2f}")

    def plot_equilibrium(self):
        # Now plot
        grid = np.linspace(2, 4, 100)
        fig, ax = plt.subplots()

        ax.plot(grid, self.qd(grid), "b-", lw=2, label="demand")
        ax.plot(grid, self.qs(grid), "g-", lw=2, label="supply")

        ax.set_xlabel("price")
        ax.set_ylabel("quantity")
        ax.legend(loc="upper center")

        plt.show()


eq = Equilibrium()
eq.compute_equilibrium()
eq.plot_equilibrium()
