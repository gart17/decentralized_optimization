class Cluster:
    
    def __init__(self, dataset, model, topology=Topology()):
        self.dataset = dataset
        self.model = model
        self.topology = topology
        self.agents = [self.generate_agent(agent_index) 
                       for agent_index in range(self.topology.n_agents)]
    
    def generate_agent(self, agent_index):
        return Agent(self.model, )
        
        
    def distribute_data(self):