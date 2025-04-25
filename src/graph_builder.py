import os
import re
import argparse
from tqdm import tqdm
import networkx as nx
from rapidfuzz import process, fuzz


def parse_bbl(bbl_path):
    """Extract reference titles from a .bbl file."""
    try:
        with open(bbl_path, 'r', encoding='utf8') as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(bbl_path, 'r', encoding='latin-1') as f:
            text = f.read()
    # split into bibitem blocks
    items = re.split(r"\\bibitem\{.*?\}", text)[1:]
    titles = []
    for block in items:
        parts = re.split(r"\\newblock", block)
        if len(parts) >= 2:
            # title is first newblock segment
            title = parts[1].strip()
            # remove trailing punctuation and braces
            title = re.sub(r"^[\{\}]+|[\{\}]+$", "", title)
            title = title.rstrip(' .')
            titles.append(title)
    return titles


def build_title_map(data_dir):
    """Map paper_id -> paper title from title.txt."""
    title_map = {}
    for paper in os.listdir(data_dir):
        pap_dir = os.path.join(data_dir, paper)
        tfile = os.path.join(pap_dir, 'title.txt')
        if os.path.isdir(pap_dir) and os.path.isfile(tfile):
            with open(tfile, 'r', encoding='utf8') as f:
                title_map[paper] = f.read().strip()
    return title_map


def build_references(data_dir):
    """Map paper_id -> list of reference titles."""
    refs = {}
    for paper in os.listdir(data_dir):
        pap_dir = os.path.join(data_dir, paper)
        if not os.path.isdir(pap_dir):
            continue
        # find .bbl
        for fname in os.listdir(pap_dir):
            if fname.endswith('.bbl'):
                bbl_path = os.path.join(pap_dir, fname)
                refs[paper] = parse_bbl(bbl_path)
                break
    return refs


def match_refs_to_papers(refs, title_map, threshold=80):
    """Fuzzy match reference titles to known paper titles."""
    graph = nx.DiGraph()
    # add nodes
    for pid in title_map:
        graph.add_node(pid)
    titles = list(title_map.values())
    pid_by_title = {v: k for k, v in title_map.items()}

    for src, rlist in tqdm(refs.items(), desc='Matching refs'):
        for ref in rlist:
            match, score, _ = process.extractOne(ref, titles, scorer=fuzz.token_set_ratio)
            if score >= threshold:
                tgt = pid_by_title.get(match)
                if tgt:
                    graph.add_edge(src, tgt)
    return graph


def main():
    parser = argparse.ArgumentParser(description='Build citation graph from .bbl files')
    parser.add_argument('--data_dir', required=True, help='Path to dataset_papers folder')
    parser.add_argument('--out', default='citation_graph.graphml', help='Output GraphML file')
    args = parser.parse_args()

    title_map = build_title_map(args.data_dir)
    refs = build_references(args.data_dir)
    G = match_refs_to_papers(refs, title_map)
    nx.write_graphml(G, args.out)
    print(f"Graph written to {args.out} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")


if __name__ == '__main__':
    main()
