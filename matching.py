import networkx as nx
from networkx.algorithms import bipartite

def optimalBipartiteMatching (costTable) :
    """
    Return the minimum cost bipartite 
    matching.

    Parameters
    ----------
    costTable : dict()
        For each pair, what is the 
        cost of matching that pair
        together.
    """
    G = nx.Graph()
    for key, val in costTable.items() :
        i, j = key
        G.add_node(i, bipartite=0)
        G.add_node(str(j), bipartite=1)
        G.add_edge(i, str(j), weight=val)
    matchingWithDups = bipartite.minimum_weight_full_matching(G)
    matching = dict()
    for key, val in matchingWithDups.items() : 
        if not isinstance(key, str):
            matching[key] = int(val)
    return matching

def bestAssignmentCost (costTable) :
    """
    Compute the minimum total
    cost assignment.

    Parameters
    ----------
    costTable : dict()
        For each pair, what is the 
        cost of matching that pair
        together.
    """
    matching = optimalBipartiteMatching(costTable)
    cost = sum([costTable[e] for e in matching.items()])
    return cost

# Differentiable Matching
def project_row(X):
    """
    p(X) = X - 1/m (X 1m - 1n) 1mˆT
    X shape: n x m
    """
    X_row_sum = X.sum(dim=1, keepdim=True) # shape n x 1
    one_m = torch.ones(1, X.shape[1]).to(X.device) # shape 1 x m
    return X - (X_row_sum - 1).mm(one_m) / X.shape[1]

def project_col(X):
    """
    p(X) = X if XˆT 1n <= 1m else X - 1/n 1n (1nˆT X - 1mˆT)
    X shape: n x m
    """
    X_col_sum = X.sum(dim=0, keepdim=True) # shape 1 x m
    one_n = torch.ones(X.shape[0], 1).to(X.device) # shape n x 1
    mask = (X_col_sum <= 1).float()
    P = X - (one_n).mm(X_col_sum - 1) / X.shape[0]
    return X * mask + (1 - mask) * P

def relax_matching(C, X_init, max_iter, proj_iter, lr):
    X = X_init
    P = [torch.zeros_like(C) for _ in range(3)]
    X_list = [X_init]

    for i in range(max_iter):
        X = X - lr * C # gradient step
        # project C onto the constrain set
        for j in range(proj_iter):
            X = X + P[0]
            Y = project_row(X)
            P[0] = X - Y

            X = Y + P[1]
            Y = project_col(X)
            P[1] = X - Y

            X = Y + P[2]
            Y = F.relu(X)
            P[2] = X - Y

            X = Y

        X_list += [X]
    return torch.stack(X_list, dim=0).mean(dim=0)
