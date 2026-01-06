import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.datasets import load_iris
from latency_aware_tree import LatencyAwareNST, encode_ttfs

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
    '''
    From: https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3
    If there is a cycle that is reachable from root, then this will see infinite recursion.
    G: the graph
    root: the root node of current branch
    width: horizontal space allocated for this branch - should remain unchanged
    vert_gap: gap between levels of hierarchy
    vert_loc: vertical location of root
    xcenter: horizontal location of root
    '''

    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = next(iter(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict that will store the positions
        parent: the parent of this node.
        '''
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx, pos=pos, 
                                    parent = root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def generate_enhanced_trace():
    # Load a simple dataset
    data = load_iris()
    X, y = data.data, data.target
    # Binarize for simpler tree if needed, but Iris classes 0, 1, 2 are fine
    
    # Train a SHALLOW tree
    T_MAX = 50
    nst = LatencyAwareNST(max_depth=3, lambda_latency=0.25, time_steps=T_MAX, n_classes=3)
    nst.fit(X, y)
    
    # Pick a sample
    sample_idx = 10  # Class 0
    x_sample = X[sample_idx]
    y_true = y[sample_idx]
    
    # Encode TTFS
    x_spikes = encode_ttfs(x_sample.reshape(1, -1), T_max=T_MAX)[0]
    
    # Build NetworkX graph
    G = nx.DiGraph()
    labels = {}
    node_colors = []
    
    # Trace the real path
    real_path = []
    node = nst.root
    while node is not None:
        real_path.append(id(node))
        if node.is_leaf:
            break
        # Simulate split
        val = x_spikes[node.feature_idx]
        if val < node.split_time:
            node = node.right
        else:
            node = node.left

    # Add nodes and edges
    def add_to_graph(node, depth=0):
        node_id = id(node)
        if node.is_leaf:
            label = f"Class {node.prediction}"
            labels[node_id] = label
            G.add_node(node_id)
            return
        
        # Internal Node
        label = f"f{node.feature_idx} < {node.split_time:.1f}"
        labels[node_id] = label
        G.add_node(node_id)
        
        # Edges
        if node.right:
            G.add_edge(node_id, id(node.right), type='Right (Early)')
            add_to_graph(node.right, depth+1)
        if node.left:
            G.add_edge(node_id, id(node.left), type='Left (Late)')
            add_to_graph(node.left, depth+1)

    add_to_graph(nst.root)
    
    # Position nodes
    pos = hierarchy_pos(G, root=id(nst.root))
    
    # Plot
    plt.figure(figsize=(14, 8))
    
    # Draw Background (Alternative) Edges
    other_edges = [e for e in G.edges() if e[0] not in real_path or e[1] not in real_path]
    nx.draw_networkx_edges(G, pos, edgelist=other_edges, edge_color='lightgray', arrows=True, width=1.0, alpha=0.5)
    
    # Draw Real Path Edges
    real_edges = []
    for i in range(len(real_path)-1):
        real_edges.append((real_path[i], real_path[i+1]))
    nx.draw_networkx_edges(G, pos, edgelist=real_edges, edge_color='#2ecc71', width=3.0, arrows=True, arrowsize=20)

    # Draw Nodes
    node_colors = ['#2ecc71' if n in real_path else '#ecf0f1' for n in G.nodes()]
    node_sizes = [3000 if n in real_path else 2000 for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors='gray')
    
    # Draw Labels
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight='bold')

    # Add Legend and Annotations
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#2ecc71', lw=4),
                   Line2D([0], [0], color='lightgray', lw=1),
                   Line2D([0], [0], marker='o', color='w', label='Current Path',
                          markerfacecolor='#2ecc71', markersize=15),
                   Line2D([0], [0], marker='o', color='w', label='Alternative',
                          markerfacecolor='#ecf0f1', markersize=12, markeredgecolor='gray')]
    
    plt.legend(custom_lines, ['Decision Path', 'Alternative Path', 'Active Node', 'Inactive Node'], loc='upper right')
    
    plt.title(f"NST Decision Trace: Real vs. Alternative Paths\nSample Prediction: {labels[real_path[-1]]} (True: Class {y_true})", fontsize=14)
    plt.axis('off')
    
    # Final cleanup and save
    plt.tight_layout()
    plt.savefig('decision_trace_enhanced.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated decision_trace_enhanced.png")

if __name__ == "__main__":
    generate_enhanced_trace()
