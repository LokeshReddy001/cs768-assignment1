#!/usr/bin/env python3
import argparse
import os
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from model import LinkPredictor

def load_graph(graph_path):
    Gnx = nx.read_graphml(graph_path)
    mapping = {nid: i for i, nid in enumerate(Gnx.nodes())}
    # build edge_index
    edges = [[mapping[u], mapping[v]] for u, v in Gnx.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index, mapping


def load_text_embeddings(data_dir, mapping, device):
    sbert = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    x = torch.zeros((len(mapping), 384), dtype=torch.float, device=device)
    for nid, idx in tqdm(mapping.items(), desc='Embedding texts'):
        txt = ''
        for fname in ('title.txt', 'abstract.txt'):
            path = os.path.join(data_dir, nid, fname)
            if os.path.isfile(path):
                txt += open(path, 'r', encoding='utf8').read().strip() + ' '
        if txt:
            emb = sbert.encode(txt, convert_to_tensor=True)
            x[idx] = emb
    return x


def main():
    parser = argparse.ArgumentParser(description='Train GraphSAGE link predictor')
    parser.add_argument('--graph', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out_model', default='models/linkpred.pt')
    args = parser.parse_args()

    device = torch.device(args.device)
    edge_index, mapping = load_graph(args.graph)
    x = load_text_embeddings(args.data_dir, mapping, device)
    edge_index = edge_index.to(device)

    # split pos edges into train/val
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges)
    val_cnt = int(0.1 * num_edges)
    val_idx = perm[:val_cnt]
    train_idx = perm[val_cnt:]
    pos_train = edge_index[:, train_idx]
    pos_val = edge_index[:, val_idx]

    model = LinkPredictor(in_dim=x.size(1), hidden=128, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs+1):
        model.train()
        optimizer.zero_grad()
        neg_train = negative_sampling(edge_index, num_neg_samples=pos_train.size(1), method='sparse')
        pos_out, neg_out = model(x, edge_index, pos_train, neg_train)
        loss = (
            F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out)) +
            F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
        ) / 2
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            neg_val = negative_sampling(edge_index, num_neg_samples=pos_val.size(1), method='sparse')
            pos_v, neg_v = model(x, edge_index, pos_val, neg_val)
            val_loss = (
                F.binary_cross_entropy_with_logits(pos_v, torch.ones_like(pos_v)) +
                F.binary_cross_entropy_with_logits(neg_v, torch.zeros_like(neg_v))
            ) / 2
        print(f'Epoch {epoch}/{args.epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}')

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    torch.save(model.state_dict(), args.out_model)
    # save final node embeddings
    emb = model.sage(x, edge_index).cpu()
    torch.save(emb, os.path.join(os.path.dirname(args.out_model), 'node_embeddings.pt'))
    print('Training complete. Model and embeddings saved.')


if __name__ == '__main__':
    main()
