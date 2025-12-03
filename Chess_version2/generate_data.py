"""
STOCKFISH DATA GENERATION - High Quality Training Data

Instead of learning from human games (which have mistakes), we'll:
1. Generate positions from varied openings
2. Use Stockfish to find the best moves
3. Add position evaluation scores
4. Create much higher quality training data

This will make the AI MUCH stronger!
"""

import os
import sys

import chess
import chess.engine
import numpy as np
import pickle
from tqdm import tqdm
import random

def board_to_feature_vector(board):
    """Convert chess board to enhanced 791-dimensional feature vector"""
    features = []
    
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    # 1. Piece positions (768 features)
    board_features = np.zeros(768, dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_idx = piece_map[piece.piece_type]
            if piece.color == chess.BLACK:
                piece_idx += 6
            board_features[piece_idx * 64 + square] = 1
    
    features.extend(board_features)
    
    # 2. Game state features (5 features)
    features.append(1.0 if board.turn == chess.WHITE else 0.0)
    features.append(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    features.append(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    features.append(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    features.append(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)
    
    # 3. Material count (12 features - 6 piece types x 2 colors)
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
        features.append(float(len(board.pieces(piece_type, chess.WHITE))))
        features.append(float(len(board.pieces(piece_type, chess.BLACK))))
    
    features.append(float(len(list(board.legal_moves))))
    
    features.append(1.0 if board.is_check() else 0.0)
    
    # Attack maps (number of squares attacked by each side)
    white_attacks = 0
    black_attacks = 0
    for square in chess.SQUARES:
        if board.is_attacked_by(chess.WHITE, square):
            white_attacks += 1
        if board.is_attacked_by(chess.BLACK, square):
            black_attacks += 1
    
    features.append(float(white_attacks))
    features.append(float(black_attacks))
    
    # Center control (how many pieces attacking center squares)
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    white_center = sum(1 for sq in center_squares if board.is_attacked_by(chess.WHITE, sq))
    black_center = sum(1 for sq in center_squares if board.is_attacked_by(chess.BLACK, sq))
    
    features.append(float(white_center))
    features.append(float(black_center))
    
    result = np.array(features, dtype=np.float32)
    
    # IMPORTANT: Verify we have exactly 791 features
    assert len(result) == 791, f"Expected 791 features, got {len(result)}"
    
    return result

def move_to_index(move):
    """Convert move to index"""
    return move.from_square * 64 + move.to_square

def generate_opening_positions(num_positions=1000):
    """Generate diverse opening positions"""
    positions = []
    
    # Common opening move sequences (first 3-6 moves)
    opening_moves = [
        ["e2e4", "e7e5", "g1f3"],  # Italian/Spanish
        ["e2e4", "c7c5"],  # Sicilian
        ["d2d4", "d7d5"],  # Queen's Pawn
        ["g1f3", "d7d5", "c2c4"],  # Reti
        ["e2e4", "e7e6"],  # French
        ["d2d4", "g8f6", "c2c4", "e7e6"],  # Nimzo-Indian
        ["e2e4", "c7c6"],  # Caro-Kann
        ["g1f3", "g8f6"],  # Various systems
    ]
    
    print("Generating opening positions...")
    for _ in tqdm(range(num_positions)):
        board = chess.Board()
        
        opening = random.choice(opening_moves)

        for move_uci in opening:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
            except:
                break

        for _ in range(random.randint(1, 3)):
            if board.is_game_over():
                break
            legal_moves = list(board.legal_moves)
            if legal_moves:
                board.push(random.choice(legal_moves))
        
        if not board.is_game_over():
            positions.append(board.copy())
    
    return positions

def generate_training_data_from_stockfish(stockfish_path, num_positions=50000, 
                                         positions_per_batch=5000, depth=15):
    
    print("="*70)
    print("STOCKFISH TRAINING DATA GENERATION")
    print("="*70)
    print(f"Target positions: {num_positions:,}")
    print(f"Stockfish depth: {depth}")
    print(f"Batch size: {positions_per_batch}")
    print("="*70 + "\n")
    
    # Initialize Stockfish
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    positions_data = []
    moves_data = []
    evals_data = []
    
    batch_num = 0
    total_generated = 0
    
    starting_positions = generate_opening_positions(num_positions // 5)
    
    print(f"\nGenerating {num_positions:,} training positions...")
    print("This will take a while but produces HIGH QUALITY data!\n")
    
    pbar = tqdm(total=num_positions, desc="Generating data")
    
    while total_generated < num_positions:
        if starting_positions:
            board = random.choice(starting_positions).copy()
        else:
            board = chess.Board()
        
        game_length = random.randint(10, 40)
        
        for move_num in range(game_length):
            if board.is_game_over():
                break
            
            try:
                result = engine.analyse(board, chess.engine.Limit(depth=depth))
                
                if 'pv' in result and len(result['pv']) > 0:
                    best_move = result['pv'][0]
                    
                    score = result.get('score', chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE))
                    
                    # Store the position, move, and evaluation
                    positions_data.append(board_to_feature_vector(board))
                    moves_data.append(move_to_index(best_move))
                    
                    # Convert score to centipawns (relative to side to move)
                    if score.relative.score(mate_score=10000) is not None:
                        eval_cp = score.relative.score(mate_score=10000)
                    else:
                        eval_cp = 0
                    evals_data.append(eval_cp)
                    
                    # Make the move
                    board.push(best_move)
                    
                    total_generated += 1
                    pbar.update(1)
                    
                    # Save batch when we reach batch size
                    if len(positions_data) >= positions_per_batch:
                        X_batch = np.array(positions_data, dtype=np.float32)
                        y_batch = np.array(moves_data, dtype=np.int32)
                        eval_batch = np.array(evals_data, dtype=np.float32)
                        
                        filename = f'stockfish_batch_{batch_num}.pkl'
                        with open(filename, 'wb') as f:
                            pickle.dump({
                                'X': X_batch,
                                'y': y_batch,
                                'evals': eval_batch
                            }, f)
                        
                        # Verify file was created
                        import os
                        if os.path.exists(filename):
                            file_size = os.path.getsize(filename) / (1024**2)
                            print(f"\n  Saved batch {batch_num}: {len(X_batch):,} positions (~{file_size:.1f} MB) at {os.path.abspath(filename)}")
                        else:
                            print(f"\n  ERROR: Failed to save {filename}")
                        
                        positions_data = []
                        moves_data = []
                        evals_data = []
                        batch_num += 1
                    
                    if total_generated >= num_positions:
                        break
            except:
                break
    
    pbar.close()
    
    # Save remaining data
    if len(positions_data) > 0:
        X_batch = np.array(positions_data, dtype=np.float32)
        y_batch = np.array(moves_data, dtype=np.int32)
        eval_batch = np.array(evals_data, dtype=np.float32)
        
        with open(f'stockfish_batch_{batch_num}.pkl', 'wb') as f:
            pickle.dump({
                'X': X_batch,
                'y': y_batch,
                'evals': eval_batch
            }, f)
        
        print(f"\n  Saved batch {batch_num}: {len(X_batch):,} positions")
        batch_num += 1
    
    # Save metadata
    metadata = {
        'num_batches': batch_num,
        'total_positions': total_generated,
        'depth': depth,
        'feature_size': 791  # Updated feature size
    }
    
    with open('stockfish_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    engine.quit()
    
    print(f"\n{'='*70}")
    print("DATA GENERATION COMPLETE!")
    print(f"{'='*70}")
    print(f"Total positions: {total_generated:,}")
    print(f"Batch files created: {batch_num}")
    print(f"Files created:")
    for i in range(batch_num):
        print(f"  - stockfish_batch_{i}.pkl")
    print(f"  - stockfish_metadata.pkl")
    print(f"\nReady for training with high-quality Stockfish data!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    stockfish_path = "stockfish.exe"  # Adjust path if needed
    
    
    generate_training_data_from_stockfish(
        stockfish_path=stockfish_path,
        num_positions=50000,  # Start with 50k, can increase later
        positions_per_batch=5000,
        depth=15  # Good balance of quality and speed
    )
    
    print("Next step")
    print("1. Run: python training.py")