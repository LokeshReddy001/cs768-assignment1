import argparse
from sentence_transformers import SentenceTransformer
import torch
import torch_geometric
import os
from torch_geometric.nn import GraphSAGE
import networkx as nx

################################################
#               IMPORTANT                      #
################################################
# 1. Do not print anything other than the ranked list of papers.
# 2. Do not forget to remove all the debug prints while submitting.

sbert = SentenceTransformer('all-MiniLM-L6-v2')

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_dim=384, hidden=128, num_layers=2):
        super().__init__()
        self.sage = GraphSAGE(in_channels=in_dim, hidden_channels=hidden, num_layers=num_layers, out_channels=hidden) # Explicitly set out_channels
        # No linear layer needed for dot product score

    def forward(self, x, edge_index):
         # Only encode nodes using GraphSAGE
        h = self.sage(x, edge_index)
        return h # Return node embeddings

    def decode(self, h, edge_label_index):
        # Decode links using dot product
        return (h[edge_label_index[0]] * h[edge_label_index[1]]).sum(dim=1)


def convert_str_to_embeddding(txt):
    max_len = 512
    if len(txt.split()) > max_len:
        txt = ' '.join(txt.split()[:max_len])
    emb = sbert.encode(txt.strip(), convert_to_tensor=True, show_progress_bar=False)
    return emb.to('cpu')

def load_full_graph_data(graph_path):
    if not os.path.exists(graph_path):
         raise FileNotFoundError(f"Graph file not found: {graph_path}")
    Gnx = nx.read_graphml(graph_path)
    # print(f"Full graph loaded: {Gnx.number_of_nodes()} nodes, {Gnx.number_of_edges()} edges.")

    nodes = list(Gnx.nodes())
    mapping = {nid: i for i, nid in enumerate(nodes)}
    inv_mapping = {i: nid for nid, i in mapping.items()}

    edges = [[mapping[u], mapping[v]] for u, v in Gnx.edges() if u in mapping and v in mapping]
    if not edges:
        print("Warning: No edges found or mapped in the graph!")
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # print(f"Full edge_index created with shape: {edge_index.shape}")
    return Gnx, edge_index



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-paper-title", type=str, required=True)
    parser.add_argument("--test-paper-abstract", type=str, required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    
    # print(args)

    ################################################
    #               YOUR CODE START                #
    ################################################
    title = args.test_paper_title
    abstract = args.test_paper_abstract
    emb = convert_str_to_embeddding(title+' '+abstract)
    # print('Loading Text Embeddings')
    w = torch.load('./models/text_embeddings.pt')

    x_full = torch.cat([w, emb.unsqueeze(0)], dim=0)
    # print('Loading Graph Data')
    G, edge_index = load_full_graph_data('./models/final_citations_graph.graphml')

    model = LinkPredictor(in_dim=x_full.size(1), hidden=64, num_layers=3).to('cpu')
    # print('Loading Model')
    model.load_state_dict(torch.load('./models/linkpred_model_final.pt', map_location='cpu'))

    full_node_emb = model(x_full, edge_index).cpu()

    nodes = list(G.nodes())
    mapping = {nid: i for i, nid in enumerate(nodes)}
    inv_mapping = {i: nid for nid, i in mapping.items()}

    query_emb = full_node_emb[-1]
    scores = torch.matmul(full_node_emb, query_emb)
    topk_scores, topk_indices_full = torch.topk(scores, k=args.k)
    
    title_map = torch.load('./models/title_map.pt')

    result = []
    for idx in topk_indices_full.tolist():
        result.append(title_map[inv_mapping[idx]])

    # result = ['paper1', 'paper2', 'paper3', 'paperK']  # Replace with your actual ranked list


    ################################################
    #               YOUR CODE END                  #
    ################################################


    
    ################################################
    #               DO NOT CHANGE                  #
    ################################################
    print('\n'.join(result))

if __name__ == "__main__":
    main()