import re
from tqdm import tqdm

def parse_pgn_file(filename, min_elo=1500):
    """Parse PGN file and filter games by ELO rating"""
    
    print(f"Reading {filename}...")
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Split into individual games
    games = content.split('\n\n[Event ')
    if games[0].startswith('[Event '):
        games[0] = games[0][7:]  # Remove '[Event ' from first game
    else:
        games = games[1:]  # Skip empty first element
    
    print(f"Found {len(games)} total games")
    
    filtered_games = []
    
    for game in tqdm(games, desc="Filtering games"):
        # Extract ELO ratings
        white_elo_match = re.search(r'\[WhiteElo "(\d+)"\]', game)
        black_elo_match = re.search(r'\[BlackElo "(\d+)"\]', game)
        
        if white_elo_match and black_elo_match:
            white_elo = int(white_elo_match.group(1))
            black_elo = int(black_elo_match.group(1))
            
            # Only keep games where BOTH players have ELO >= min_elo
            if white_elo >= min_elo and black_elo >= min_elo:
                filtered_games.append('[Event ' + game)
    
    print(f"\nFiltered to {len(filtered_games)} games with both players ELO >= {min_elo}")
    
    # Write filtered games to new file
    output_filename = f'filtered_games_elo{min_elo}.pgn'
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(filtered_games))
    
    print(f"Saved to {output_filename}")
    return output_filename

if __name__ == "__main__":
    input_file = "chess.pgn"
    min_elo = 1500
    
    output_file = parse_pgn_file(input_file, min_elo)
    print(f"\nDone! Filtered games saved to: {output_file}")