import matplotlib.pyplot as plt
import networkx as nx

def analyze_graph(graph):
    print("Number of edges:", graph.number_of_edges())
    print("Number of isolated nodes:", nx.number_of_isolates(graph))
    degrees = [d for _, d in graph.degree()]
    in_degrees = [d for _, d in graph.in_degree()]
    out_degrees = [d for _, d in graph.out_degree()]
    # print("Average degree:", sum(degrees)/len(degrees))
    print("Average in-degree:", sum(in_degrees)/len(in_degrees))
    print("Average out-degree:", sum(out_degrees)/len(out_degrees))

    # print(degrees)
    plt.hist(degrees, bins=1000)
    plt.title("Degree Histogram")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.xlim(0, 100) 
    # plt.show()
    plt.savefig("degree_histogram1.png")

    # return
    if nx.is_connected(graph.to_undirected()):
        print("Diameter of graph:", nx.diameter(graph.to_undirected()))
    else:
        largest_cc = max(nx.connected_components(graph.to_undirected()), key=len)
        subgraph = graph.subgraph(largest_cc)
        print("Diameter of largest connected component:", nx.diameter(subgraph.to_undirected()))

G = nx.read_graphml('./models/final_citations_graph.graphml')
analyze_graph(G)