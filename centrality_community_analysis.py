# Enhanced Centrality and Community Analysis for Citation Networks
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import community.community_louvain as community_louvain
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CitationNetworkAnalyzer:
    """
    Comprehensive citation network analyzer with centrality and community detection
    """
    
    def __init__(self, papers_file='papers.csv', edges_file='edges.csv'):
        self.papers_file = papers_file
        self.edges_file = edges_file
        self.G = None
        self.papers_dict = {}
        self.centrality_measures = {}
        self.communities = {}
        self.analysis_results = {}
        
    def load_data(self):
        """Load and preprocess citation network data"""
        try:
            print("Loading citation network data...")
            
            # Load data files
            df = pd.read_csv(self.papers_file)
            de = pd.read_csv(self.edges_file)
            
            # Preprocessing: Remove papers with null abstracts
            null_ids = df[df['abstract'].isnull()]['id']
            de = de[~de['source'].isin(null_ids) & ~de['target'].isin(null_ids)]
            df = df[~df['id'].isin(null_ids)]
            
            # Create graph
            self.G = nx.from_pandas_edgelist(de, 'source', 'target', create_using=nx.DiGraph())
            
            # Add node attributes
            unique_df = df.drop_duplicates(subset='id', keep='first')
            self.papers_dict = unique_df.set_index('id').to_dict('index')
            nx.set_node_attributes(self.G, self.papers_dict)
            
            print(f"✓ Graph loaded successfully:")
            print(f"  - Nodes: {self.G.number_of_nodes()}")
            print(f"  - Edges: {self.G.number_of_edges()}")
            print(f"  - Density: {nx.density(self.G):.4f}")
            print(f"  - Average degree: {sum(dict(self.G.degree()).values()) / self.G.number_of_nodes():.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def calculate_centrality_measures(self):
        """Calculate various centrality measures"""
        print("\nCalculating centrality measures...")
        
        try:
            # Basic centrality measures
            self.centrality_measures['degree'] = nx.degree_centrality(self.G)
            self.centrality_measures['betweenness'] = nx.betweenness_centrality(self.G)
            self.centrality_measures['closeness'] = nx.closeness_centrality(self.G)
            
            # Advanced centrality measures
            try:
                self.centrality_measures['eigenvector'] = nx.eigenvector_centrality_numpy(self.G)
            except:
                print("  ⚠️  Eigenvector centrality failed (possibly disconnected graph)")
                self.centrality_measures['eigenvector'] = {node: 0 for node in self.G.nodes()}
            
            # PageRank centrality
            self.centrality_measures['pagerank'] = nx.pagerank(self.G)
            
            # Katz centrality
            try:
                self.centrality_measures['katz'] = nx.katz_centrality(self.G)
            except:
                print("  ⚠️  Katz centrality failed")
                self.centrality_measures['katz'] = {node: 0 for node in self.G.nodes()}
            
            print("✓ Centrality measures calculated successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error calculating centrality measures: {str(e)}")
            return False
    
    def detect_communities(self):
        """Detect communities using multiple algorithms"""
        print("\nDetecting communities...")
        
        try:
            # Louvain community detection
            self.communities['louvain'] = community_louvain.best_partition(self.G.to_undirected())
            
            # Label propagation
            try:
                self.communities['label_propagation'] = nx.community.label_propagation_communities(self.G.to_undirected())
            except:
                print("  ⚠️  Label propagation failed")
                self.communities['label_propagation'] = []
            
            # Girvan-Newman (for smaller components)
            try:
                components = list(nx.connected_components(self.G.to_undirected()))
                gn_communities = []
                for component in components:
                    if len(component) > 3:
                        subgraph = self.G.to_undirected().subgraph(component)
                        communities = nx.community.girvan_newman(subgraph)
                        gn_communities.extend(list(next(communities)))
                self.communities['girvan_newman'] = gn_communities
            except:
                print("  ⚠️  Girvan-Newman algorithm failed")
                self.communities['girvan_newman'] = []
            
            print("✓ Community detection completed")
            return True
            
        except Exception as e:
            print(f"❌ Error detecting communities: {str(e)}")
            return False
    
    def create_visualizations(self, save_plots=True):
        """Create comprehensive visualizations"""
        print("\nCreating visualizations...")
        
        try:
            # Set up the plotting style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Create figure with subplots
            fig = plt.figure(figsize=(24, 16))
            
            # Layout for visualizations
            pos = nx.spring_layout(self.G, k=1, iterations=50, seed=42)
            
            # 1. Degree Centrality
            plt.subplot(3, 4, 1)
            node_sizes = [self.centrality_measures['degree'][node] * 3000 + 50 for node in self.G.nodes()]
            nx.draw(self.G, pos, node_size=node_sizes, node_color='lightblue', 
                   edge_color='gray', alpha=0.7, arrows=False)
            plt.title('Degree Centrality\n(Node size ∝ centrality)', fontweight='bold')
            plt.axis('off')
            
            # 2. Betweenness Centrality
            plt.subplot(3, 4, 2)
            node_colors = [self.centrality_measures['betweenness'][node] for node in self.G.nodes()]
            nx.draw(self.G, pos, node_size=100, node_color=node_colors, 
                   edge_color='gray', alpha=0.7, arrows=False, cmap=plt.cm.Reds)
            plt.title('Betweenness Centrality\n(Color intensity ∝ centrality)', fontweight='bold')
            plt.axis('off')
            
            # 3. Closeness Centrality
            plt.subplot(3, 4, 3)
            node_colors = [self.centrality_measures['closeness'][node] for node in self.G.nodes()]
            nx.draw(self.G, pos, node_size=100, node_color=node_colors, 
                   edge_color='gray', alpha=0.7, arrows=False, cmap=plt.cm.Blues)
            plt.title('Closeness Centrality\n(Color intensity ∝ centrality)', fontweight='bold')
            plt.axis('off')
            
            # 4. Eigenvector Centrality
            plt.subplot(3, 4, 4)
            node_colors = [self.centrality_measures['eigenvector'][node] for node in self.G.nodes()]
            nx.draw(self.G, pos, node_size=100, node_color=node_colors, 
                   edge_color='gray', alpha=0.7, arrows=False, cmap=plt.cm.Greens)
            plt.title('Eigenvector Centrality\n(Color intensity ∝ centrality)', fontweight='bold')
            plt.axis('off')
            
            # 5. PageRank Centrality
            plt.subplot(3, 4, 5)
            node_colors = [self.centrality_measures['pagerank'][node] for node in self.G.nodes()]
            nx.draw(self.G, pos, node_size=100, node_color=node_colors, 
                   edge_color='gray', alpha=0.7, arrows=False, cmap=plt.cm.Purples)
            plt.title('PageRank Centrality\n(Color intensity ∝ centrality)', fontweight='bold')
            plt.axis('off')
            
            # 6. Louvain Communities
            plt.subplot(3, 4, 6)
            num_communities = len(set(self.communities['louvain'].values()))
            colors = plt.cm.Set3(np.linspace(0, 1, num_communities))
            node_colors = [colors[self.communities['louvain'][node]] for node in self.G.nodes()]
            nx.draw(self.G, pos, node_size=100, node_color=node_colors, 
                   edge_color='gray', alpha=0.7, arrows=False)
            plt.title(f'Louvain Communities\n({num_communities} communities)', fontweight='bold')
            plt.axis('off')
            
            # 7. Combined Centrality (Degree + Betweenness)
            plt.subplot(3, 4, 7)
            combined_centrality = {}
            for node in self.G.nodes():
                combined_centrality[node] = (self.centrality_measures['degree'][node] + 
                                          self.centrality_measures['betweenness'][node]) / 2
            node_sizes = [combined_centrality[node] * 2000 + 50 for node in self.G.nodes()]
            node_colors = [combined_centrality[node] for node in self.G.nodes()]
            nx.draw(self.G, pos, node_size=node_sizes, node_color=node_colors, 
                   edge_color='gray', alpha=0.7, arrows=False, cmap=plt.cm.Oranges)
            plt.title('Combined Centrality\n(Degree + Betweenness)', fontweight='bold')
            plt.axis('off')
            
            # 8. Network Statistics
            plt.subplot(3, 4, 8)
            plt.axis('off')
            stats_text = f"""
Network Statistics:
• Nodes: {self.G.number_of_nodes()}
• Edges: {self.G.number_of_edges()}
• Density: {nx.density(self.G):.4f}
• Avg Degree: {sum(dict(self.G.degree()).values()) / self.G.number_of_nodes():.2f}
• Communities: {num_communities}
• Connected Components: {nx.number_connected_components(self.G.to_undirected())}
            """
            plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
            # 9. Centrality Distribution (Degree)
            plt.subplot(3, 4, 9)
            degree_values = list(self.centrality_measures['degree'].values())
            plt.hist(degree_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Degree Centrality Distribution', fontweight='bold')
            plt.xlabel('Degree Centrality')
            plt.ylabel('Frequency')
            
            # 10. Centrality Distribution (Betweenness)
            plt.subplot(3, 4, 10)
            betweenness_values = list(self.centrality_measures['betweenness'].values())
            plt.hist(betweenness_values, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
            plt.title('Betweenness Centrality Distribution', fontweight='bold')
            plt.xlabel('Betweenness Centrality')
            plt.ylabel('Frequency')
            
            # 11. Community Size Distribution
            plt.subplot(3, 4, 11)
            community_sizes = {}
            for node, comm in self.communities['louvain'].items():
                if comm not in community_sizes:
                    community_sizes[comm] = 0
                community_sizes[comm] += 1
            sizes = list(community_sizes.values())
            plt.hist(sizes, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            plt.title('Community Size Distribution', fontweight='bold')
            plt.xlabel('Community Size')
            plt.ylabel('Frequency')
            
            # 12. Centrality Correlation
            plt.subplot(3, 4, 12)
            nodes = list(self.G.nodes())
            degree_vals = [self.centrality_measures['degree'][node] for node in nodes]
            betweenness_vals = [self.centrality_measures['betweenness'][node] for node in nodes]
            plt.scatter(degree_vals, betweenness_vals, alpha=0.6, color='purple')
            plt.title('Degree vs Betweenness Centrality', fontweight='bold')
            plt.xlabel('Degree Centrality')
            plt.ylabel('Betweenness Centrality')
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig('centrality_community_analysis.png', dpi=300, bbox_inches='tight')
                print("✓ Visualizations saved as 'centrality_community_analysis.png'")
            
            plt.show()
            return True
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {str(e)}")
            return False
    
    def generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        print("\nGenerating analysis report...")
        
        try:
            # Calculate statistics
            stats = {
                'network_stats': {
                    'nodes': self.G.number_of_nodes(),
                    'edges': self.G.number_of_edges(),
                    'density': nx.density(self.G),
                    'average_degree': sum(dict(self.G.degree()).values()) / self.G.number_of_nodes(),
                    'connected_components': nx.number_connected_components(self.G.to_undirected())
                },
                'centrality_stats': {},
                'community_stats': {}
            }
            
            # Centrality statistics
            for measure_name, centrality_dict in self.centrality_measures.items():
                values = list(centrality_dict.values())
                stats['centrality_stats'][measure_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
            
            # Community statistics
            community_sizes = {}
            for node, comm in self.communities['louvain'].items():
                if comm not in community_sizes:
                    community_sizes[comm] = 0
                community_sizes[comm] += 1
            
            stats['community_stats'] = {
                'num_communities': len(set(self.communities['louvain'].values())),
                'community_sizes': community_sizes,
                'largest_community': max(community_sizes.values()),
                'smallest_community': min(community_sizes.values()),
                'avg_community_size': np.mean(list(community_sizes.values()))
            }
            
            # Find top central nodes
            top_nodes = {}
            for measure_name, centrality_dict in self.centrality_measures.items():
                top_nodes[measure_name] = sorted(centrality_dict.items(), 
                                               key=lambda x: x[1], reverse=True)[:10]
            
            # Generate report
            report = {
                'timestamp': datetime.now().isoformat(),
                'statistics': stats,
                'top_central_nodes': top_nodes,
                'community_analysis': self._analyze_communities()
            }
            

            return report
            
        except Exception as e:
            print(f"❌ Error generating report: {str(e)}")
            return None
    
    def _analyze_communities(self):
        """Analyze community characteristics"""
        community_analysis = {}
        
        for comm_id in set(self.communities['louvain'].values()):
            community_nodes = [node for node, comm in self.communities['louvain'].items() if comm == comm_id]
            
            if len(community_nodes) > 1:
                # Find most central node in this community
                comm_centrality = {node: self.centrality_measures['degree'][node] for node in community_nodes}
                most_central = max(comm_centrality.items(), key=lambda x: x[1])
                
                # Get paper information
                paper_info = {}
                if most_central[0] in self.papers_dict:
                    paper_info = {
                        'title': self.papers_dict[most_central[0]].get('title', 'Unknown'),
                        'authors': self.papers_dict[most_central[0]].get('authorships', 'Unknown'),
                        'topic': self.papers_dict[most_central[0]].get('topic', 'Unknown')
                    }
                
                community_analysis[f'community_{comm_id}'] = {
                    'size': len(community_nodes),
                    'most_central_node': most_central[0],
                    'centrality_score': most_central[1],
                    'paper_info': paper_info
                }
        
        return community_analysis
    
    def print_summary(self):
        """Print analysis summary"""
        print("\n" + "="*60)
        print("CITATION NETWORK ANALYSIS SUMMARY")
        print("="*60)
        
        # Network statistics
        print(f"\n📊 NETWORK STATISTICS:")
        print(f"  • Nodes: {self.G.number_of_nodes()}")
        print(f"  • Edges: {self.G.number_of_edges()}")
        print(f"  • Density: {nx.density(self.G):.4f}")
        print(f"  • Average degree: {sum(dict(self.G.degree()).values()) / self.G.number_of_nodes():.2f}")
        print(f"  • Connected components: {nx.number_connected_components(self.G.to_undirected())}")
        
        # Centrality statistics
        print(f"\n🎯 CENTRALITY STATISTICS:")
        for measure_name, centrality_dict in self.centrality_measures.items():
            values = list(centrality_dict.values())
            print(f"  • {measure_name.title()}:")
            print(f"    - Max: {np.max(values):.4f}")
            print(f"    - Min: {np.min(values):.4f}")
            print(f"    - Mean: {np.mean(values):.4f}")
        
        # Community statistics
        community_sizes = {}
        for node, comm in self.communities['louvain'].items():
            if comm not in community_sizes:
                community_sizes[comm] = 0
            community_sizes[comm] += 1
        
        print(f"\n🏘️  COMMUNITY STATISTICS:")
        print(f"  • Number of communities: {len(set(self.communities['louvain'].values()))}")
        print(f"  • Largest community: {max(community_sizes.values())} nodes")
        print(f"  • Smallest community: {min(community_sizes.values())} nodes")
        print(f"  • Average community size: {np.mean(list(community_sizes.values())):.1f} nodes")
        
        # Top central nodes
        print(f"\n⭐ TOP CENTRAL NODES:")
        for measure_name in ['degree', 'betweenness', 'closeness']:
            if measure_name in self.centrality_measures:
                top_nodes = sorted(self.centrality_measures[measure_name].items(), 
                                 key=lambda x: x[1], reverse=True)[:3]
                print(f"  • Top 3 by {measure_name.title()} Centrality:")
                for node, centrality in top_nodes:
                    title = self.papers_dict.get(node, {}).get('title', 'Unknown')[:50] + "..."
                    print(f"    {node}: {centrality:.4f} - {title}")
        
        print("\n" + "="*60)
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        print("🚀 Starting comprehensive citation network analysis...")
        
        # Load data
        if not self.load_data():
            return False
        
        # Calculate centrality measures
        if not self.calculate_centrality_measures():
            return False
        
        # Detect communities
        if not self.detect_communities():
            return False
        
        # Create visualizations
        if not self.create_visualizations():
            return False
        
        # Generate report
        report = self.generate_analysis_report()
        
        # Print summary
        self.print_summary()
        
        print("\n✅ Analysis completed successfully!")
        print("📁 Generated files:")
        print("  • centrality_community_analysis.png")
        
        return True

def main():
    """Main function to run the analysis"""
    analyzer = CitationNetworkAnalyzer()
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n🎉 Analysis completed successfully!")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")

if __name__ == "__main__":
    main()