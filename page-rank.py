import networkx as nx


def pagerank(G, alpha=0.85,  max_iter=100, tol=1.0e-6, nstart=None, weight='weight'):
    """

    :param G:
    :param alpha:
    :param personalization:
    :param max_iter:
    :param tol:
    :param nstart:
    :param weight:
    :param dangling:
    :return:
    """

    if len(G) == 0:
        return {}

    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G

    # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()

    # Choose fixed starting vector if not given
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        # Normalized nstart vector
        s = float(sum(nstart.values()))
        x = dict((k, v / s) for k, v in nstart.items())


    p = dict.fromkeys(W, 1.0 / N)

    dangling_weights = p
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

    # Power iteration: mate up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)

        for n in x:

            for nbr in W[n]:
                x[nbr] += alpha * xlast[n] * W[n][nbr][weight]
            x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n]

        # Check convergence
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N*tol:
            return x
    return x