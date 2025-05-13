# visualize_simple.py
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load data with error handling
try:
    articles_df = pd.read_csv("articles.csv", encoding='utf-8-sig')
    relations_df = pd.read_csv("relations.csv", encoding='utf-8-sig', 
                             header=0,
                             names=['source_id', 'target_id'],
                             on_bad_lines='warn')
except pd.errors.ParserError as e:
    print(f"CSV parsing error: {e}")
    exit()

# Create directed graph
G = nx.DiGraph()

# Add nodes with enhanced styling
for _, row in articles_df.iterrows():
    G.add_node(row['id'],
              label=f"Art {row['id']}",
              title=row['title'][:50] + "...",
              depth=row['id'])  # Simple depth based on ID

# Reverse edge direction to show "cites" instead of "cited by"
for _, row in relations_df.iterrows():
    G.add_edge(row['target_id'], row['source_id'])  # Inverted direction

# Improved layout configuration
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

# Custom node styling
node_colors = [G.nodes[n].get('depth', 0) for n in G.nodes()]
node_sizes = [800 + 200 * G.degree(n) for n in G.nodes()]

# Draw elements
nx.draw_networkx_nodes(G, pos,
                       node_color=node_colors,
                       cmap='viridis',
                       node_size=node_sizes,
                       alpha=0.9,
                       edgecolors='black')

nx.draw_networkx_edges(G, pos,
                       edge_color='#FF5722',
                       width=1.2,
                       arrowsize=25,
                       arrowstyle='->,head_width=0.4,head_length=0.4')

nx.draw_networkx_labels(G, pos,
                        font_size=9,
                        font_weight='bold',
                        font_family='sans-serif')

# Add legend and annotations
plt.title("Article Citation Graph (Direction: Cites →)", fontsize=14)
plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), 
             label='Article Depth', shrink=0.95)

# Add explanatory text
plt.text(0.5, -0.1, 
         "Arrows show citation direction: Article A → Article B means A cites B",
         ha='center', va='center', transform=plt.gca().transAxes)

plt.axis('off')
plt.tight_layout()

# Save and show
plt.savefig('citation_graph.png', dpi=300, bbox_inches='tight')
plt.show()