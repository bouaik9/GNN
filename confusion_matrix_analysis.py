import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import negative_sampling
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def load_and_preprocess_data():
    """Load and preprocess the citation network data"""
    print("Loading and preprocessing data...")
    
    # Load data
    df = pd.read_csv('papers.csv')
    de = pd.read_csv('edges.csv')
    
    # Preprocessing: Remove papers with null abstracts
    null_ids = df[df['abstract'].isnull()]['id']
    de = de[~de['source'].isin(null_ids) & ~de['target'].isin(null_ids)]
    df = df[~df['id'].isin(null_ids)]
    
    # Encode node IDs
    node_encoder = LabelEncoder()
    all_nodes = list(set(de['source']).union(set(de['target'])))
    node_encoder.fit(all_nodes)
    
    source = node_encoder.transform(de['source'])
    target = node_encoder.transform(de['target'])
    edge_index = torch.tensor(np.vstack([source, target]), dtype=torch.long)
    
    # Extract TF-IDF features
    texts = df['abstract'].fillna('')
    vectorizer = TfidfVectorizer(max_features=300)
    tfidf_features = vectorizer.fit_transform(texts).toarray()
    
    id_to_index = {pid: idx for idx, pid in enumerate(df['id'])}
    
    features = []
    for node_id in node_encoder.classes_:
        idx = id_to_index.get(node_id)
        if idx is not None:
            features.append(tfidf_features[idx])
        else:
            features.append(np.zeros(tfidf_features.shape[1]))
    features = np.array(features)
    
    x = torch.tensor(features, dtype=torch.float)
    
    # Create PyG data object
    data = Data(x=x, edge_index=edge_index)
    
    # Split edges into train/val/test
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=False,
        split_labels=True,
        add_negative_train_samples=True,
    )
    train_data, val_data, test_data = transform(data)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return train_data, val_data, test_data, device, node_encoder, vectorizer

# Model Definitions
class LinkPredictor(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.lin1 = torch.nn.Linear(emb_dim * 2, 64)
        self.lin2 = torch.nn.Linear(64, 1)

    def forward(self, z, edge_index):
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        x = torch.cat([src, dst], dim=1)
        x = F.relu(self.lin1(x))
        x = torch.sigmoid(self.lin2(x)).squeeze()
        return x

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class SAGEEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.sage1 = SAGEConv(in_channels, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = self.sage2(x, edge_index)
        return x

class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x

def train_model(model, predictor, train_data, val_data, device, epochs=100):
    """Train a model and return training history"""
    model.train()
    predictor.train()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=0.01)
    
    history = {'loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Get embeddings
        z = model(train_data.x.to(device), train_data.edge_index.to(device))
        
        # Positive and negative edges
        pos_edge_index = train_data.pos_edge_label_index.to(device)
        neg_edge_index = train_data.neg_edge_label_index.to(device)
        
        # If no negatives, sample manually
        if neg_edge_index is None or neg_edge_index.size(1) == 0:
            neg_edge_index = negative_sampling(
                edge_index=train_data.edge_index.to(device),
                num_nodes=train_data.num_nodes,
                num_neg_samples=pos_edge_index.size(1),
            )
        
        # Get predictions
        pos_pred = predictor(z, pos_edge_index)
        neg_pred = predictor(z, neg_edge_index)
        
        # Labels
        pos_labels = torch.ones(pos_pred.size(0), device=device)
        neg_labels = torch.zeros(neg_pred.size(0), device=device)
        
        # Combined predictions and labels
        preds = torch.cat([pos_pred, neg_pred], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)
        
        # Loss
        loss = F.binary_cross_entropy(preds, labels)
        loss.backward()
        optimizer.step()
        
        # Validation
        if epoch % 10 == 0:
            model.eval()
            predictor.eval()
            with torch.no_grad():
                z = model(val_data.x.to(device), val_data.edge_index.to(device))
                val_pos_pred = predictor(z, val_data.pos_edge_label_index.to(device))
                val_neg_pred = predictor(z, val_data.neg_edge_label_index.to(device))
                
                val_preds = torch.cat([val_pos_pred, val_neg_pred], dim=0)
                val_labels = torch.cat([torch.ones(val_pos_pred.size(0), device=device),
                                      torch.zeros(val_neg_pred.size(0), device=device)], dim=0)
                
                val_acc = ((val_preds > 0.5) == val_labels).float().mean().item()
                print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}')
                
                history['loss'].append(loss.item())
                history['val_acc'].append(val_acc)
            
            model.train()
            predictor.train()
    
    return history

def evaluate_model(model, predictor, test_data, device):
    """Evaluate a model and return detailed metrics"""
    model.eval()
    predictor.eval()
    
    with torch.no_grad():
        z = model(test_data.x.to(device), test_data.edge_index.to(device))
        
        # Test predictions
        test_pos_pred = predictor(z, test_data.pos_edge_label_index.to(device))
        test_neg_pred = predictor(z, test_data.neg_edge_label_index.to(device))
        
        test_preds = torch.cat([test_pos_pred, test_neg_pred], dim=0).cpu().numpy()
        test_labels = torch.cat([torch.ones(test_pos_pred.size(0), device=device),
                               torch.zeros(test_neg_pred.size(0), device=device)], dim=0).cpu().numpy()
        
        # Convert to binary predictions
        test_preds_binary = (test_preds > 0.5).astype(int)
        
        # Calculate metrics
        cm = confusion_matrix(test_labels, test_preds_binary)
        roc_auc = roc_auc_score(test_labels, test_preds)
        
        return {
            'confusion_matrix': cm,
            'predictions': test_preds,
            'labels': test_labels,
            'roc_auc': roc_auc,
            'binary_predictions': test_preds_binary
        }

def create_confusion_matrix_visualization(results):
    """Create comprehensive confusion matrix visualization for all models"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Comprehensive Model Comparison: Confusion Matrices and ROC Curves', 
                 fontsize=16, fontweight='bold')
    
    # Colors for each model
    colors = {'GraphSAGE': 'blue', 'GCN': 'green', 'GAT': 'red'}
    
    for idx, (model_name, result) in enumerate(results.items()):
        # Confusion Matrix
        ax1 = axes[0, idx]
        cm = result['confusion_matrix']
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Link', 'Link'],
                   yticklabels=['No Link', 'Link'],
                   ax=ax1)
        ax1.set_title(f'{model_name} Confusion Matrix', fontweight='bold')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Add metrics text
        metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # ROC Curve
        ax2 = axes[1, idx]
        fpr, tpr, _ = roc_curve(result['labels'], result['predictions'])
        roc_auc = result['roc_auc']
        
        ax2.plot(fpr, tpr, color=colors[model_name], lw=2, 
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title(f'{model_name} ROC Curve', fontweight='bold')
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_model_comparison_table(results):
    """Create a detailed comparison table for all models"""
    print("\n" + "="*80)
    print("DETAILED MODEL COMPARISON TABLE")
    print("="*80)
    
    comparison_data = []
    
    for model_name, result in results.items():
        cm = result['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        comparison_data.append({
            'Model': model_name,
            'True Positives': tp,
            'True Negatives': tn,
            'False Positives': fp,
            'False Negatives': fn,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Specificity': specificity,
            'ROC AUC': result['roc_auc']
        })
    
    # Create formatted table
    print(f"{'Model':<12} {'TP':<4} {'TN':<4} {'FP':<4} {'FN':<4} {'Acc':<6} {'Prec':<6} {'Rec':<6} {'F1':<6} {'Spec':<6} {'AUC':<6}")
    print("-" * 80)
    
    for data in comparison_data:
        print(f"{data['Model']:<12} {data['True Positives']:<4} {data['True Negatives']:<4} "
              f"{data['False Positives']:<4} {data['False Negatives']:<4} "
              f"{data['Accuracy']:<6.3f} {data['Precision']:<6.3f} {data['Recall']:<6.3f} "
              f"{data['F1-Score']:<6.3f} {data['Specificity']:<6.3f} {data['ROC AUC']:<6.3f}")
    
    # Find best model for each metric
    print("\n" + "="*50)
    print("BEST PERFORMING MODEL BY METRIC")
    print("="*50)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'ROC AUC']
    for metric in metrics:
        best_model = max(comparison_data, key=lambda x: x[metric])
        print(f"{metric:<12}: {best_model['Model']} ({best_model[metric]:.3f})")
    
    return comparison_data

def main():
    """Main function to run the comprehensive confusion matrix analysis"""
    print("Starting comprehensive confusion matrix analysis for all models...")
    
    # Load and preprocess data
    train_data, val_data, test_data, device, node_encoder, vectorizer = load_and_preprocess_data()
    
    # Define the three models
    models = {
        'GraphSAGE': SAGEEncoder(in_channels=train_data.x.size(1), hidden_channels=64, out_channels=32),
        'GCN': GCNEncoder(in_channels=train_data.x.size(1), hidden_channels=64, out_channels=32),
        'GAT': GATEncoder(in_channels=train_data.x.size(1), hidden_channels=64, out_channels=32, heads=2)
    }
    
    # Store results for each model
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training and Evaluating {model_name}")
        print(f"{'='*50}")
        
        # Move model to device
        model = model.to(device)
        predictor = LinkPredictor(emb_dim=32).to(device)
        
        # Train model
        history = train_model(model, predictor, train_data, val_data, device)
        
        # Evaluate model
        result = evaluate_model(model, predictor, test_data, device)
        results[model_name] = result
        
        # Print results
        print(f"\n{model_name} Results:")
        print(f"ROC AUC: {result['roc_auc']:.4f}")
        print(f"Confusion Matrix:")
        print(result['confusion_matrix'])
        print(f"\nClassification Report:")
        print(classification_report(result['labels'], result['binary_predictions'], 
                                 target_names=['No Link', 'Link']))
    
    # Create comprehensive visualization
    create_confusion_matrix_visualization(results)
    
    # Create detailed comparison table
    comparison_data = create_model_comparison_table(results)
    
    print("\nAnalysis complete! Check 'confusion_matrices_comparison.png' for visualizations.")
    
    return results, comparison_data

if __name__ == "__main__":
    results, comparison_data = main() 