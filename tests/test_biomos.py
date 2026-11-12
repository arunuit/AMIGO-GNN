import networkx as nx
import numpy as np
from src.biomos import BioMOS

def test_biomos_returns_nonempty_feature_mask():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 5))
    y = (X[:, 0] > 0).astype(int)
    names = [f"f{i}" for i in range(X.shape[1])]
    graph = nx.path_graph(names)
    opt = BioMOS(population_size=6, iterations=3, seed=7)
    opt.fit(X, y, graph, names)
    assert opt.mask_.shape == (5,)
    assert opt.mask_.any()
