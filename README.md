# Chess AI

A lightweight neural network-based chess engine trained on game data. This project implements a chess AI using PyTorch that can play against humans through a visual interface.

## Overview

This project contains two versions of a chess AI:

- **Version 1**: Trained on human game data from PGN files
- **Version 2** (Chess_version2): Trained on Stockfish-generated positions for higher quality

**Note:** These models are intentionally lightweight and designed for web deployment. They are not tournament-strength engines and can be beaten by intermediate players. The focus is on making legal, reasonable moves with minimal computational requirements.

## Features

- Neural network move prediction (773-791 input features)
- Visual gameplay interface using Pygame
- Multi-task learning (move prediction + position evaluation)
- Batch training system for handling large datasets
- Model checkpointing and resumable training

## Requirements

```bash
pip install torch numpy chess pygame python-chess tqdm
```

For data generation (Version 2 only):
- Stockfish chess engine

## Project Structure

### Root Directory (Version 1)
- `extract_features.py` - Converts PGN games to feature vectors
- `filter_best_games.py` - Filters games by player ELO rating
- `train_model.py` - Trains the basic model
- `play_visual.py` - Visual chess game interface
- `chess_model.pth` - Trained model weights

### Chess_version2 Directory (Advanced Version)
- `generate_data.py` - Generates training data using Stockfish
- `training.py` - Advanced multi-task training
- `play_visual.py` - Improved visual interface with evaluation display
- `chess_model_advanced.pth` - Advanced model weights
- `chess_model_checkpoint.pth` - Training checkpoint
- `chess_model.onnx` - ONNX format for web deployment
- `conversion.py` - Converts PyTorch models to ONNX

## Getting Started

### Version 1: Training from Human Games

1. **Download chess game data**
   - Get PGN files from [Lichess Database](https://database.lichess.org/) or [FICS Games Database](https://www.ficsgames.org/download.html)
   - Rename the downloaded file to `chess.pgn`

2. **Filter high-quality games**
   ```bash
   python filter_best_games.py
   ```
   This creates `filtered_games_elo1500.pgn` with games from players rated 1500+

3. **Extract features**
   ```bash
   python extract_features.py
   ```
   Converts games into training batches (batch_0.pkl, batch_1.pkl, etc.)

4. **Train the model**
   ```bash
   python train_model.py
   ```
   Trains for 15 epochs. Model saved as `chess_model.pth`

5. **Play against the AI**
   ```bash
   python play_visual.py
   ```

### Version 2: Training with Stockfish (Recommended)

1. **Download Stockfish**
   - Get from [Stockfish official website](https://stockfishchess.org/download/)
   - Place `stockfish.exe` (Windows) or `stockfish` (Linux/Mac) in Chess_version2 directory

2. **Generate training data**
   ```bash
   cd Chess_version2
   python generate_data.py
   ```
   Generates 50,000 positions evaluated by Stockfish. This takes several hours.

3. **Train the advanced model**
   ```bash
   python training.py
   ```
   Trains for 10 epochs with multi-task learning. Run multiple times to continue training.

4. **Play against the AI**
   ```bash
   python play_visual.py
   ```

## How It Works

### Feature Representation
The board state is converted to a feature vector:
- 768 features: Piece positions (12 piece types × 64 squares)
- 5 features: Game state (turn, castling rights)
- Additional features (Version 2): Material count, attack maps, center control

### Model Architecture
- Version 1: 3-layer feedforward network (1024→512→256 neurons)
- Version 2: Multi-task network with shared layers and dual heads:
  - Move prediction head (4096 possible moves)
  - Position evaluation head (centipawn score)

### Training
- Input: Board position features
- Output: Probability distribution over all possible moves
- Loss: Cross-entropy for moves, MSE for evaluation
- Optimizer: Adam/AdamW with learning rate scheduling

## Playing the Game

Controls:
- Click pieces to select and move
- R key: Reset game
- ESC key: Quit

The AI plays as Black. The interface shows:
- Current game state
- Move count
- Position evaluation (Version 2)
- Check/checkmate warnings

## Model Performance

These models are designed for lightweight deployment and educational purposes. Performance characteristics:

- Move legality: 100% (only legal moves are played)
- Playing strength: Approximately 1200-1400 ELO
- Response time: < 1 second per move on CPU
- Model size: ~15-25 MB

The models will make reasonable opening moves and basic tactics but can be easily beaten by a beginner player.

## Training Tips

1. **More data = better performance**: Generate 100k+ positions for stronger play
2. **Increase Stockfish depth**: Higher depth (20-25) gives better training data but takes longer
3. **Train longer**: 20-30 epochs with early stopping often works well
4. **Use validation accuracy**: Stop when validation accuracy plateaus
5. **Resume training**: Both versions support checkpoint resuming

## File Formats

- `.pkl` - Pickle files containing training batches
- `.pth` - PyTorch model weights
- `.onnx` - ONNX format for web/cross-platform deployment
- `.pgn` - Standard chess game notation files

## Limitations

- Not suitable for competitive play
- Limited opening book knowledge
- Weak endgame play compared to traditional engines
- No opening database or tablebase support
- Tactical depth limited by network capacity

## Future Improvements

- Increase model size and training data
- Add opening book
- Implement Monte Carlo Tree Search (MCTS)
- Train on GPU clusters for stronger play
- Add difficulty levels
- Implement time controls

## Acknowledgments

- Chess library: [python-chess](https://github.com/niklasf/python-chess)
- Training data: Lichess game database
- Engine analysis: Stockfish chess engine
- UI: Pygame library
