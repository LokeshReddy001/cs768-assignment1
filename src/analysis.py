#!/usr/bin/env python3
import argparse
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Compute metrics on citation graph.')
    parser.add_argument('--graph', required=True, help='Path to GraphML file')
    parser.add_argument('--hist', default='degree_histogram.png', help='Output path for degree histogram')
    args = parser.parse_args()

    # Load graph
    G = nx.read_graphml(args.graph)

    # Basic stats
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    isolates = list(nx.isolates(G))
    num_isolates = len(isolates)

    # Degree stats
    if G.is_directed():
        in_degs = [d for _, d in G.in_degree()]
        out_degs = [d for _, d in G.out_degree()]
        comps = nx.weakly_connected_components(G)
        largest = max(comps, key=len)
        subG = G.subgraph(largest).to_undirected()
    else:
        in_degs = out_degs = [d for _, d in G.degree()]
        comps = nx.connected_components(G)
        largest = max(comps, key=len)
        subG = G.subgraph(largest)

    avg_in = np.mean(in_degs)
    avg_out = np.mean(out_degs)
    degrees = [d for _, d in G.degree()]

    # Diameter
    try:
        diam = nx.diameter(subG)
    except Exception:
        diam = None

    # Output
    print(f'Number of nodes: {num_nodes}')
    print(f'Number of edges: {num_edges}')
    print(f'Number of isolated nodes: {num_isolates}')
    print(f'Average in-degree: {avg_in:.2f}')
    print(f'Average out-degree: {avg_out:.2f}')
    print(f'Diameter of largest component: {diam}')

    # Plot degree histogram
    plt.figure(figsize=(8,6))
    plt.hist(degrees, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    plt.title('Degree Distribution')
    plt.tight_layout()
    plt.savefig(args.hist)
    print(f'Degree histogram saved to {args.hist}')

if __name__ == '__main__':
    main()
