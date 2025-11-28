import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gc

class ChessDataset(Dataset):
    def __init__(self, batch_files, indices):
        """Dataset that loads from batch files on demand"""
        self.batch_files = batch_files
        self.indices = indices  # List of (batch_idx, position_idx) tuples
        self.current_batch_idx = None
        self.current_batch = None
    
    def _load_batch(self, batch_idx):
        """Load a batch file, keeping only one in memory at a time"""
        if batch_idx != self.current_batch_idx:
            # Clear previous batch
            self.current_batch = None
            gc.collect()
            
            # Load new batch
            with open(self.batch_files[batch_idx], 'rb') as f:
                self.current_batch = pickle.load(f)
            self.current_batch_idx = batch_idx
        
        return self.current_batch
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        batch_idx, pos_idx = self.indices[idx]
        batch_data = self._load_batch(batch_idx)
        
        X = torch.FloatTensor(batch_data['X'][pos_idx])
        y = torch.LongTensor([batch_data['y'][pos_idx]])[0]
        
        return X, y

class ChessNet(nn.Module):
    def __init__(self, input_size=773, hidden_sizes=[1024, 512, 256], output_size=4096):
        super(ChessNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def create_indices(num_batches):
    """Create list of all (batch_idx, position_idx) pairs"""
    indices = []
    
    print("Loading batch information...")
    for batch_idx in tqdm(range(num_batches), desc="Reading batches"):
        with open(f'batch_{batch_idx}.pkl', 'rb') as f:
            batch_data = pickle.load(f)
            batch_size = len(batch_data['X'])
            
            for pos_idx in range(batch_size):
                indices.append((batch_idx, pos_idx))
        
        # Clear memory after reading
        del batch_data
        gc.collect()
    
    return indices

def train_model(num_batches, epochs=15, batch_size=256, learning_rate=0.001):
    """Train the chess neural network using batch files"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get all batch files
    batch_files = [f'batch_{i}.pkl' for i in range(num_batches)]
    
    # Create indices for all positions
    print("\nPreparing dataset...")
    indices = create_indices(num_batches)
    print(f"Total positions: {len(indices):,}")
    
    # Sort indices by batch for better memory efficiency
    print("Sorting indices for memory efficiency...")
    indices.sort(key=lambda x: x[0])
    
    # Split into train/val
    split_idx = int(0.9 * len(indices))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    print(f"Training samples: {len(train_indices):,}")
    print(f"Validation samples: {len(val_indices):,}")
    
    # Create datasets
    train_dataset = ChessDataset(batch_files, train_indices)
    val_dataset = ChessDataset(batch_files, val_indices)
    
    # Use smaller batch size and no workers to save memory
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize smaller model
    model = ChessNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_val_loss = float('inf')
    best_val_acc = 0
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for i, (batch_X, batch_y) in enumerate(train_bar):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
            
            train_bar.set_postfix({
                'loss': f'{train_loss/(i+1):.4f}',
                'acc': f'{100*train_correct/train_total:.2f}%'
            })
            
            # Clear cache periodically
            if i % 100 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for i, (batch_X, batch_y) in enumerate(val_bar):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
                
                val_bar.set_postfix({
                    'loss': f'{val_loss/(i+1):.4f}',
                    'acc': f'{100*val_correct/val_total:.2f}%'
                })
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        scheduler.step(avg_val_loss)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {100*train_correct/train_total:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'chess_model.pth')
            print(f"  ✓ New best model saved! (Acc: {val_acc:.2f}%)")
        print()
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return model

if __name__ == "__main__":
    # Load metadata
    with open('batch_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    num_batches = metadata['num_batches']
    total_positions = metadata['total_positions']
    
    print(f"Found {num_batches} batch files with {total_positions:,} positions")
    
    print("\nUsing smaller model and batch size to reduce memory usage.\n")
    
    model = train_model(num_batches, epochs=15, batch_size=256, learning_rate=0.001)
    
    print("\nTraining complete! Model saved to chess_model.pth")