#!/usr/bin/env python3
import argparse
import torch
import networkx as nx
from model import LinkPredictor

def load_graph(graph_path):
    G = nx.read_graphml(graph_path)
    mapping = {nid: i for i, nid in enumerate(G.nodes())}
    inv_map = {i: nid for nid, i in mapping.items()}
    return G, mapping, inv_map

def main():
    parser = argparse.ArgumentParser(description='Inference for link prediction')
    parser.add_argument('--graph', required=True, help='GraphML path')
    parser.add_argument('--emb', required=True, help='Path to node embeddings (pt)')
    parser.add_argument('--model', required=True, help='Path to trained model (pt)')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden size used in model')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GraphSAGE layers')
    parser.add_argument('--k', type=int, default=5, help='Top-K predictions')
    args = parser.parse_args()

    G, mapping, inv_map = load_graph(args.graph)
    # Load embeddings and linear head parameters
    emb = torch.load(args.emb)
    state = torch.load(args.model, map_location='cpu')
    lin_weight = state['lin.weight']
    lin_bias = state['lin.bias']

    # Test nodes = those with at least one outgoing edge
    test_nodes = [nid for nid in G.nodes() if G.out_degree(nid) > 0]
    # Limit to first 10 for demo
    for nid in test_nodes[:10]:
        u = mapping[nid]
        # Build pairwise feature [h_u || h_i]
        h_u = emb[u].unsqueeze(0)
        h_u_rep = h_u.repeat(emb.size(0), 1)
        pair = torch.cat([h_u_rep, emb], dim=1)
        # Score with linear head
        scores = (pair @ lin_weight.t()).view(-1) + lin_bias
        # exclude self
        scores[u] = -float('inf')
        topk = torch.topk(scores, args.k).indices.tolist()
        preds = [inv_map[idx] for idx in topk]
        actual = set(G.successors(nid))
        hit = any(p in actual for p in preds)
        print(f'Node {nid}: Predicted={preds}, Actual={list(actual)}, Hit={hit}')

if __name__ == '__main__':
    main()
