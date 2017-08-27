class Topology:
    
    def __init__(self, n_agent=5, density=0.6, policy='metropolis', adjacency_matrix=None):
        if adjacency_matrix is not None:
            assert(adjacency_matrix.shape[0] == adjacency_matrix.shape[1])
            self.adjacency_matrix = adjacency_matrix
            self.n_agent = adjacency_matrix.shape[0]
        else:
            graph = nx.random_geometric_graph(n_agent, radius=density)
            if policy == 'metropolis':
                adjacency_matrix = combination_policy.metropolis_matrix(graph)
            elif policy == 'average':
                adjacency_matrix = combination_policy.averaging_matrix(graph)             
            else:
                print('Combination policy {0} not supported yet.'.format(rule))
                raise NotImplementedError
            self.adjacency_matrix = adjacency_matrix
            self.n_agent = n_agent
        
    def neighbor_weights(self, node_index):
        return self.adjacency_matrix[node_index, :]