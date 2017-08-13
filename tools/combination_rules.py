def averaging_matrix(G, nodelist=None, weight='weight'):
    '''Return the averaging matrix of G.
    
    The Row l, Column k of the average matrix A of a graph G
    is given by A_{lk} = 1/n_k, if l is a neighbor of k; and
    0 otherwise. 
    
    Parameters
    ----------
    G : graph
        A NetworkX graph.
        
    nodelist : list, optional 
        The rows and columns are ordered according to the nodes in nodelist.
        If nodelist is None, then the ordering is produced by G.nodes().
        
    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.
    
    Returns
    -------
    A : Scipy sparse matrix
        The averaging matrix of G.
        
    References
    ----------
    .. [1] Ali H. Sayed (2014), 
        "Adaptation, Learning, and Optimization over Networks", 
        Foundations and Trends in Machine Learning: Vol. 7: No. 4-5, p 664. 
        http://dx.doi.org/10.1561/2200000051
    '''
    if nodelist is None:
        nodelist = list(G)
        
    Adj = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, 
                weight=weight, format='csr').toarray()
    if np.all(np.diag(Adj) == 0):  # add self-loops
        np.fill_diagonal(Adj, 1) 
    
    return Adj * (1.0 / np.sum(Adj, axis=0))




