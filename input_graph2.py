# visualize_simple.py
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load data with robust error handling
try:
    articles_df = pd.read_csv("articles.csv", encoding='utf-8-sig')
    relations_df = pd.read_csv("relations.csv", encoding='utf-8-sig',
                             header=0,
                             names=['source_id', 'target_id'],
                             on_bad_lines='warn')
except pd.errors.ParserError as e:
    print(f"CSV parsing error: {e}")
    print("Please check your CSV files for formatting issues!")
    exit()

# Create directed graph
G = nx.DiGraph()

# Add nodes with enhanced metadata
for _, row in articles_df.iterrows():
    G.add_node(row['id'],
              label=f"{row['id']}",
              title=row['title'],
              depth=row['id'])

# Add reversed edges to show "CITES" relationship
for _, row in relations_df.iterrows():
    G.add_edge(row['target_id'], row['source_id'])  # Inverted direction

# Create figure with explicit axes
fig, ax = plt.subplots(figsize=(14, 10))

# Calculate layout
pos = nx.spring_layout(G, k=0.6, iterations=100, seed=42)

# Style calculations
node_colors = [G.nodes[n]['depth'] for n in G.nodes]
node_sizes = [800 + 300 * G.in_degree(n) for n in G.nodes]  # Size by citation count

# Draw nodes
nodes = nx.draw_networkx_nodes(
    G, pos, ax=ax,
    node_color=node_colors,
    cmap='plasma',
    node_size=node_sizes,
    alpha=0.95,
    edgecolors='gray',
    linewidths=0.5
)

# Draw edges
nx.draw_networkx_edges(
    G, pos, ax=ax,
    edge_color='#FF6F00',
    width=1.5,
    arrowsize=30,
    arrowstyle='->,head_width=0.6,head_length=0.6',
    connectionstyle='arc3,rad=0.1'
)

# Draw labels
nx.draw_networkx_labels(
    G, pos, ax=ax,
    labels={n: G.nodes[n]['label'] for n in G.nodes},
    font_size=10,
    font_weight='bold',
    font_family='sans-serif',
    alpha=0.9
)

# Create colorbar properly
sm = plt.cm.ScalarMappable(
    cmap='plasma',
    norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors))
)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.9, pad=0.02)
cbar.set_label('Article Generation Depth', rotation=270, labelpad=20)

# Add annotations
ax.set_title("Paper Citation Network Visualization\n(Arrows show citation direction: A → B = A cites B)",
            fontsize=14, pad=20)
ax.text(0.5, -0.05,
        "Node size represents number of citations received\nNode color indicates scraping depth from origin article",
        ha='center', va='center', transform=ax.transAxes,
        fontsize=10)

plt.axis('off')
plt.tight_layout()

# Save and show
plt.savefig('citation_network.png', dpi=300, bbox_inches='tight')
plt.show()