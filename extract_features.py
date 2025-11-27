import chess
import chess.pgn
import numpy as np
import pickle
from tqdm import tqdm
import os

def board_to_feature_vector(board):
    """Convert chess board to 773-dimensional feature vector"""
    features = []
    
    # 1. Piece positions (768 features: 12 piece types x 64 squares)
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    board_features = np.zeros(768)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_idx = piece_map[piece.piece_type]
            if piece.color == chess.BLACK:
                piece_idx += 6
            board_features[piece_idx * 64 + square] = 1
    
    features.extend(board_features)
    
    # 2. Turn (1 feature)
    features.append(1 if board.turn == chess.WHITE else 0)
    
    # 3. Castling rights (4 features)
    features.append(1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
    features.append(1 if board.has_queenside_castling_rights(chess.WHITE) else 0)
    features.append(1 if board.has_kingside_castling_rights(chess.BLACK) else 0)
    features.append(1 if board.has_queenside_castling_rights(chess.BLACK) else 0)
    
    return np.array(features, dtype=np.float32)

def move_to_index(move):
    """Convert move to index (0-4095 for all possible from-to combinations)"""
    from_square = move.from_square
    to_square = move.to_square
    return from_square * 64 + to_square

def extract_positions_from_pgn(pgn_file, max_games=200000, positions_per_batch=50000):
    """Extract board positions and moves from PGN file in small batches"""
    
    positions = []
    moves = []
    batch_num = 0
    total_positions = 0
    
    with open(pgn_file, 'r', encoding='utf-8', errors='ignore') as f:
        game_count = 0
        
        with tqdm(total=max_games, desc="Extracting positions") as pbar:
            while game_count < max_games:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                board = game.board()
                for move in game.mainline_moves():
                    # Store position before move
                    positions.append(board_to_feature_vector(board))
                    moves.append(move_to_index(move))
                    
                    # Make the move
                    board.push(move)
                    
                    # Save batch when we reach positions_per_batch positions
                    if len(positions) >= positions_per_batch:
                        X_batch = np.array(positions, dtype=np.float32)
                        y_batch = np.array(moves, dtype=np.int32)
                        
                        with open(f'batch_{batch_num}.pkl', 'wb') as bf:
                            pickle.dump({'X': X_batch, 'y': y_batch}, bf)
                        
                        total_positions += len(positions)
                        print(f"\n  Saved batch {batch_num}: {len(positions)} positions (~{len(positions)*773*4/(1024**2):.1f} MB)")
                        
                        positions = []
                        moves = []
                        batch_num += 1
                
                game_count += 1
                pbar.update(1)
    
    # Save remaining positions
    if len(positions) > 0:
        X_batch = np.array(positions, dtype=np.float32)
        y_batch = np.array(moves, dtype=np.int32)
        
        with open(f'batch_{batch_num}.pkl', 'wb') as bf:
            pickle.dump({'X': X_batch, 'y': y_batch}, bf)
        
        total_positions += len(positions)
        print(f"\n  Saved batch {batch_num}: {len(positions)} positions (~{len(positions)*773*4/(1024**2):.1f} MB)")
        batch_num += 1
    
    print(f"\nExtracted {total_positions} positions from {game_count} games")
    print(f"Saved in {batch_num} batch files")
    
    # Create metadata file
    metadata = {
        'num_batches': batch_num,
        'total_positions': total_positions,
        'total_games': game_count
    }
    with open('batch_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("\nBatch files created:")
    for i in range(min(5, batch_num)):
        print(f"  - batch_{i}.pkl")
    if batch_num > 5:
        print(f"  ... and {batch_num - 5} more")
    print("  - batch_metadata.pkl")
    
    return batch_num, total_positions

if __name__ == "__main__":
    pgn_file = "filtered_games_elo1500.pgn"
    
    print("Extracting features from games in small batches...")
    print("Each batch will contain ~50,000 positions (~150 MB)\n")
    
    num_batches, total_pos = extract_positions_from_pgn(pgn_file, max_games=200000, positions_per_batch=50000)
    
    print(f"\nDone! Created {num_batches} batch files with {total_pos:,} total positions")
    print("Ready for training!")