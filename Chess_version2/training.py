"""
ADVANCED TRAINING - Multi-Task Learning

This trains on BOTH:
1. Move prediction (what move to make)
2. Position evaluation (how good is the position)

This dual-task approach makes the AI understand chess much better
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gc

class StockfishDataset(Dataset):
    def __init__(self, batch_files, indices):
        self.batch_files = batch_files
        self.indices = indices
        self.current_batch_idx = None
        self.current_batch = None
    
    def _load_batch(self, batch_idx):
        if batch_idx != self.current_batch_idx:
            self.current_batch = None
            gc.collect()
            
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
        eval_score = torch.FloatTensor([batch_data['evals'][pos_idx]])[0]
        
        return X, y, eval_score

class AdvancedChessNet(nn.Module):
    def __init__(self, input_size=791, output_size=4096):
        super(AdvancedChessNet, self).__init__()
        
        # Shared feature extractor - OPTIMIZED (smaller but efficient)
        self.shared = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
        )
        
        # Move prediction head
        self.move_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_size)
        )
        
        # Position evaluation head
        self.eval_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        move_logits = self.move_head(shared_features)
        eval_score = self.eval_head(shared_features)
        return move_logits, eval_score

def create_indices(num_batches):
    indices = []
    
    print("Loading batch information and checking feature dimensions...")
    for batch_idx in tqdm(range(num_batches), desc="Reading batches"):
        with open(f'stockfish_batch_{batch_idx}.pkl', 'rb') as f:
            batch_data = pickle.load(f)
            batch_size = len(batch_data['X'])
            
            # Check feature size
            if batch_idx == 0:
                feature_size = batch_data['X'].shape[1]
                print(f"\nDetected feature size: {feature_size}")
                if feature_size != 791:
                    print(f"⚠ WARNING: Expected 791 features, got {feature_size}")
                    print(f"This data was generated with old feature extraction.")
                    print(f"\nPlease regenerate data by running:")
                    print(f"  python 8_generate_stockfish_data.py")
                    raise ValueError(f"Feature mismatch: expected 791, got {feature_size}")
            
            for pos_idx in range(batch_size):
                indices.append((batch_idx, pos_idx))
        
        del batch_data
        gc.collect()
    
    return indices

def train_advanced_model(num_batches, epochs=10, batch_size=1024, learning_rate=0.001, resume=True):
    """Train the advanced multi-task chess model - OPTIMIZED FOR SPEED"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    
    # Set torch to use multiple CPU threads
    torch.set_num_threads(4)
    
    batch_files = [f'stockfish_batch_{i}.pkl' for i in range(num_batches)]
    
    print("\nPreparing dataset...")
    indices = create_indices(num_batches)
    print(f"Total positions: {len(indices):,}")
    
    print("Shuffling data...")
    np.random.shuffle(indices)
    
    split_idx = int(0.9 * len(indices))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    print(f"Training samples: {len(train_indices):,}")
    print(f"Validation samples: {len(val_indices):,}")
    
    train_dataset = StockfishDataset(batch_files, train_indices)
    val_dataset = StockfishDataset(batch_files, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=0, pin_memory=False)
    
    # Initialize model
    model = AdvancedChessNet().to(device)
    
    # Two loss functions (multi-task)
    move_criterion = nn.CrossEntropyLoss()
    eval_criterion = nn.MSELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Try to resume from checkpoint
    start_epoch = 0
    best_val_acc = 0
    
    if resume and os.path.exists('chess_model_advanced.pth'):
        print("\n" + "="*70)
        print("FOUND EXISTING MODEL - RESUMING TRAINING")
        print("="*70)
        
        checkpoint = torch.load('chess_model_advanced.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
        
        if 'val_acc' in checkpoint:
            best_val_acc = checkpoint['val_acc']
            print(f"Previous best validation accuracy: {best_val_acc:.2f}%")
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Restored optimizer state")
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Restored scheduler state")
        
        print("="*70 + "\n")
    else:
        print(f"\nStarting fresh training (no checkpoint found)")
    
    print(f"\nAdvanced Model Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\n{'='*70}")
    print("STARTING ADVANCED MULTI-TASK TRAINING")
    print("Learning BOTH move selection AND position evaluation")
    print(f"Training for {epochs} epochs (current session)")
    print(f"{'='*70}\n")
    
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        train_move_loss = 0
        train_eval_loss = 0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for i, (batch_X, batch_y, batch_eval) in enumerate(train_bar):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            batch_eval = batch_eval.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            
            # Forward pass - get both outputs
            move_logits, eval_pred = model(batch_X)
            
            # Calculate both losses
            loss_move = move_criterion(move_logits, batch_y)
            loss_eval = eval_criterion(eval_pred, batch_eval / 100.0)  
            
            # Combined loss (move prediction is more important)
            total_loss = loss_move + 0.1 * loss_eval
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_move_loss += loss_move.item()
            train_eval_loss += loss_eval.item()
            
            _, predicted = torch.max(move_logits.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
            
            train_bar.set_postfix({
                'move_loss': f'{train_move_loss/(i+1):.4f}',
                'eval_loss': f'{train_eval_loss/(i+1):.4f}',
                'acc': f'{100*train_correct/train_total:.2f}%'
            })
        
        # Validation
        model.eval()
        val_move_loss = 0
        val_eval_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
            for i, (batch_X, batch_y, batch_eval) in enumerate(val_bar):
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                batch_eval = batch_eval.to(device).unsqueeze(1)
                
                move_logits, eval_pred = model(batch_X)
                
                loss_move = move_criterion(move_logits, batch_y)
                loss_eval = eval_criterion(eval_pred, batch_eval / 100.0)
                
                val_move_loss += loss_move.item()
                val_eval_loss += loss_eval.item()
                
                _, predicted = torch.max(move_logits.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
                
                val_bar.set_postfix({
                    'move_loss': f'{val_move_loss/(i+1):.4f}',
                    'eval_loss': f'{val_eval_loss/(i+1):.4f}',
                    'acc': f'{100*val_correct/val_total:.2f}%'
                })
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        scheduler.step()
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train: Move Acc={train_acc:.2f}%, Move Loss={train_move_loss/len(train_loader):.4f}, "
              f"Eval Loss={train_eval_loss/len(train_loader):.4f}")
        print(f"  Val:   Move Acc={val_acc:.2f}%, Move Loss={val_move_loss/len(val_loader):.4f}, "
              f"Eval Loss={val_eval_loss/len(val_loader):.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Always save checkpoint (for resuming)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'val_acc': val_acc,
            'feature_size': 791
        }
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, 'chess_model_advanced.pth')
            print(f"  ✓ NEW BEST MODEL! Validation Accuracy: {val_acc:.2f}%")
            print(f"  Saved to: chess_model_advanced.pth")
        else:
            # Save current state to different file so we can resume
            torch.save(checkpoint, 'chess_model_checkpoint.pth')
            print(f"  Checkpoint saved (not better than best: {best_val_acc:.2f}%)")
        
        print(f"{'='*70}\n")
        
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\n{'='*70}")
    print("TRAINING SESSION COMPLETE!")
    print(f"{'='*70}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: chess_model_advanced.pth")
    print(f"\nTo continue training, just run this script again")
    print(f"It will automatically resume from epoch {epoch+2}")
    print(f"{'='*70}\n")
    
    return model

if __name__ == "__main__":
    print("="*70)
    print("ADVANCED CHESS AI TRAINING - OPTIMIZED")
    
    # Load metadata
    with open('stockfish_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    
    num_batches = metadata['num_batches']
    total_positions = metadata['total_positions']
    
    print(f"Found {num_batches} batch files with {total_positions:,} Stockfish positions\n")
    
    model = train_advanced_model(num_batches, epochs=10, batch_size=1024, learning_rate=0.001)
    
    print("\n✓ Training complete!")
    print("\nNext step:")
    print("Run this script again to train 10 more epochs (it will auto-resume)")
    print("Keep training until validation accuracy stops improving")