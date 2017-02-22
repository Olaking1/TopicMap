import networkx as nx

def network_efficiency(G, n):
    """Computes the efficiency of ``G``.

        Parameters
        ----------
        G : NetworkX graph

        Returns
        -------
        efficiency : Integer
    """
    n = G.number_of_nodes()
    dict_shortest_paths = nx.all_pairs_shortest_path_length(G)
    sum = 0
    shortest_paths = dict_shortest_paths.values()
    for item in shortest_paths:
        for d in item.values():
            if d is not 0:
                sum += 1/d

    return sum/(n*(n-1))
