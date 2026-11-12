from __future__ import annotations
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class BioMOS:
    """
    Binary graph-aware feature selection.

    Fitness:
        F = alpha*A + beta*Gc + gamma*R - delta*D

    A  : validation accuracy from a lightweight classifier
    Gc : retained graph connectivity
    R  : centrality-weighted biological relevance
    D  : average absolute feature correlation

    The manuscript does not provide a machine-readable exact optimizer update
    equation, so this implementation follows the algorithmic steps stated in
    Algorithm 1: random binary population, local/global best, graph guidance,
    redundancy removal and iterative replacement.
    """
    def __init__(self, population_size=50, iterations=100, alpha=.40, beta=.25,
                 gamma=.20, delta=.15, graph_awareness=.30, eval_epochs=5,
                 seed=42):
        self.population_size = population_size
        self.iterations = iterations
        self.alpha, self.beta, self.gamma, self.delta = alpha, beta, gamma, delta
        self.graph_awareness = graph_awareness
        self.eval_epochs = eval_epochs
        self.rng = np.random.default_rng(seed)
        self.history = []

    def _connectivity(self, mask, graph, feature_names):
        if mask.sum() == 0: return 0.0
        selected = {feature_names[i] for i,v in enumerate(mask) if v}
        degrees = []
        for node, d in graph.degree():
            if str(node) in selected:
                degrees.append(d)
        if not degrees:
            return 0.0
        return float(np.mean(degrees) / max(1, graph.number_of_nodes()-1))

    def _relevance(self, mask, graph, feature_names):
        if mask.sum() == 0: return 0.0
        pr = nx.pagerank(graph) if graph.number_of_nodes() else {}
        vals = [pr.get(feature_names[i], 0.0) for i,v in enumerate(mask) if v]
        if not vals: return 0.0
        mx = max(pr.values()) if pr else 1.0
        return float(np.mean(vals) / max(mx, 1e-12))

    def _redundancy(self, X, mask):
        idx = np.flatnonzero(mask)
        if len(idx) < 2: return 0.0
        c = np.corrcoef(X[:, idx], rowvar=False)
        tri = np.triu_indices_from(c, 1)
        return float(np.mean(np.abs(c[tri]))) if len(tri[0]) else 0.0

    def _accuracy(self, X, y, mask):
        idx = np.flatnonzero(mask)
        if len(idx) == 0: return 0.0
        clf = LogisticRegression(max_iter=300, random_state=0)
        n = len(y)
        cut = max(1, int(.8*n))
        clf.fit(X[:cut, idx], y[:cut])
        pred = clf.predict(X[cut:, idx])
        return float(accuracy_score(y[cut:], pred))

    def fit(self, X, y, graph, feature_names):
        d = X.shape[1]
        pop = self.rng.integers(0, 2, size=(self.population_size, d), dtype=np.int8)
        pop[:, self.rng.integers(0, d, size=self.population_size)] = 1
        scores = np.zeros(self.population_size)
        local_best = pop.copy()
        local_scores = np.full(self.population_size, -np.inf)
        global_best, global_score = None, -np.inf

        for it in range(self.iterations):
            for i in range(self.population_size):
                m = pop[i]
                A = self._accuracy(X, y, m)
                Gc = self._connectivity(m, graph, feature_names)
                R = self._relevance(m, graph, feature_names)
                D = self._redundancy(X, m)
                F = self.alpha*A + self.beta*Gc + self.gamma*R - self.delta*D
                scores[i] = F
                if F > local_scores[i]:
                    local_scores[i], local_best[i] = F, m.copy()
                if F > global_score:
                    global_score, global_best = F, m.copy()

            self.history.append(float(global_score))

            # Binary swarm-like update guided by local best, global best,
            # and graph-aware stochastic perturbations.
            for i in range(self.population_size):
                p = pop[i].astype(float)
                r1, r2 = self.rng.random(d), self.rng.random(d)
                guidance = (global_best - p) * self.graph_awareness
                local = r1 * (local_best[i] - p)
                global_term = r2 * (global_best - p)
                prob = 1 / (1 + np.exp(-(local + global_term + guidance)))
                pop[i] = (self.rng.random(d) < prob).astype(np.int8)
                if pop[i].sum() == 0:
                    pop[i, self.rng.integers(0,d)] = 1

        self.mask_ = global_best.astype(bool)
        self.best_fitness_ = float(global_score)
        return self

    def transform(self, X):
        return X[:, self.mask_]

    def save_history(self, path):
        import pandas as pd
        pd.DataFrame({"iteration": np.arange(1, len(self.history)+1),
                      "best_fitness": self.history}).to_csv(path, index=False)
