import chess
import torch
import torch.nn as nn
import numpy as np
import pygame
import sys

class AdvancedChessNet(nn.Module):
    """
    Multi-task network matching the training code
    """
    def __init__(self, input_size=791, output_size=4096):
        super(AdvancedChessNet, self).__init__()
        
        # Shared feature extractor
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

class ChessAI:
    def __init__(self, model_path='chess_model_advanced.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model on {self.device}...")
        
        # Load model with advanced architecture
        self.model = AdvancedChessNet().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                print(f"Loaded model from epoch {checkpoint['epoch'] + 1}")
            if 'val_acc' in checkpoint:
                print(f"Model accuracy: {checkpoint['val_acc']:.2f}%")
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def board_to_feature_vector(self, board):
        """
        Enhanced feature extraction (791 features total)
        Matches the training data generation
        """
        features = []
        
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        # 1. Piece positions (768 features: 12 piece types × 64 squares)
        board_features = np.zeros(768)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_idx = piece_map[piece.piece_type]
                if piece.color == chess.BLACK:
                    piece_idx += 6
                board_features[piece_idx * 64 + square] = 1
        features.extend(board_features)
        
        # 2. Basic game state (5 features)
        features.append(1 if board.turn == chess.WHITE else 0)
        features.append(1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
        features.append(1 if board.has_queenside_castling_rights(chess.WHITE) else 0)
        features.append(1 if board.has_kingside_castling_rights(chess.BLACK) else 0)
        features.append(1 if board.has_queenside_castling_rights(chess.BLACK) else 0)
        
        # 3. En passant (1 feature)
        features.append(1 if board.ep_square is not None else 0)
        
        # 4. Check status (1 feature)
        features.append(1 if board.is_check() else 0)
        
        # 5. Material count (12 features: 6 piece types × 2 colors)
        for color in [chess.WHITE, chess.BLACK]:
            for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                             chess.ROOK, chess.QUEEN, chess.KING]:
                count = len(board.pieces(piece_type, color))
                features.append(count / 8.0)  # Normalize
        
        # 6. Attack maps (2 features: squares attacked by each side)
        white_attacks = len([sq for sq in chess.SQUARES if board.is_attacked_by(chess.WHITE, sq)])
        black_attacks = len([sq for sq in chess.SQUARES if board.is_attacked_by(chess.BLACK, sq)])
        features.append(white_attacks / 64.0)
        features.append(black_attacks / 64.0)
        
        # Total: 768 + 5 + 1 + 1 + 12 + 2 = 789 features
        # Add 2 more padding features to reach 791
        features.extend([0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def index_to_move(self, index):
        from_square = index // 64
        to_square = index % 64
        return chess.Move(from_square, to_square)
    
    def get_best_move(self, board, top_k=200):
        """Find best legal move, considering checks and checkmates"""
        features = self.board_to_feature_vector(board)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get both move predictions and position evaluation
            move_logits, eval_score = self.model(features_tensor)
            probabilities = torch.softmax(move_logits, dim=1)
            top_moves = torch.topk(probabilities, top_k)
            
            # Store evaluation for display
            self.last_eval = eval_score.item() * 100.0  # Denormalize
        
        legal_moves = list(board.legal_moves)
        
        # Try to find a legal move from top predictions
        for idx in top_moves.indices[0]:
            move = self.index_to_move(idx.item())
            
            # Check all legal moves for matches (including promotion)
            for legal_move in legal_moves:
                if legal_move.from_square == move.from_square and legal_move.to_square == move.to_square:
                    # Make sure the move is actually legal (handles check/checkmate)
                    test_board = board.copy()
                    try:
                        test_board.push(legal_move)
                        return legal_move
                    except:
                        continue
        
        # Fallback: pick best legal move based on simple heuristics
        # Prioritize: checkmate > check > capture > other
        checkmate_moves = []
        check_moves = []
        capture_moves = []
        other_moves = []
        
        for move in legal_moves:
            test_board = board.copy()
            test_board.push(move)
            
            if test_board.is_checkmate():
                checkmate_moves.append(move)
            elif test_board.is_check():
                check_moves.append(move)
            elif board.is_capture(move):
                capture_moves.append(move)
            else:
                other_moves.append(move)
        
        # Return best move by priority
        if checkmate_moves:
            return checkmate_moves[0]
        elif check_moves:
            import random
            return random.choice(check_moves)
        elif capture_moves:
            import random
            return random.choice(capture_moves)
        else:
            import random
            return random.choice(other_moves) if other_moves else legal_moves[0]

class ChessGUI:
    def __init__(self):
        pygame.init()
        
        # Constants - REDUCED SIZE to fit screen
        self.SQUARE_SIZE = 60  # Reduced from 80
        self.BOARD_SIZE = 8 * self.SQUARE_SIZE
        self.INFO_HEIGHT = 140  # Reduced from 200
        self.WIDTH = self.BOARD_SIZE
        self.HEIGHT = self.BOARD_SIZE + self.INFO_HEIGHT
        
        # Colors
        self.WHITE_SQUARE = (240, 217, 181)
        self.BLACK_SQUARE = (181, 136, 99)
        self.HIGHLIGHT = (255, 255, 0, 128)
        self.SELECTED = (50, 205, 50, 128)
        self.BG_COLOR = (44, 62, 80)
        self.TEXT_COLOR = (255, 255, 255)
        
        # Setup
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Advanced Chess AI - Visual Game")
        
        self.board = chess.Board()
        self.ai = ChessAI()
        self.selected_square = None
        self.player_color = chess.WHITE
        
        # Fonts
        self.font_medium = pygame.font.Font(None, 30)  # Reduced from 36
        self.font_small = pygame.font.Font(None, 22)   # Reduced from 28
        self.font_tiny = pygame.font.Font(None, 18)    # Reduced from 22
        
        # Create piece images
        self.piece_images = self.create_piece_images()
        
        self.message = "Your turn (White)"
        self.ai_thinking = False
    
    def create_piece_images(self):
        """Create simple graphical representations of chess pieces"""
        pieces = {}
        size = self.SQUARE_SIZE - 10
        
        piece_data = {
            'P': ('pawn', chess.WHITE),
            'N': ('knight', chess.WHITE),
            'B': ('bishop', chess.WHITE),
            'R': ('rook', chess.WHITE),
            'Q': ('queen', chess.WHITE),
            'K': ('king', chess.WHITE),
            'p': ('pawn', chess.BLACK),
            'n': ('knight', chess.BLACK),
            'b': ('bishop', chess.BLACK),
            'r': ('rook', chess.BLACK),
            'q': ('queen', chess.BLACK),
            'k': ('king', chess.BLACK)
        }
        
        for symbol, (piece_type, color) in piece_data.items():
            surface = pygame.Surface((size, size), pygame.SRCALPHA)
            piece_color = (255, 255, 255) if color == chess.WHITE else (50, 50, 50)
            outline_color = (0, 0, 0) if color == chess.WHITE else (200, 200, 200)
            
            center_x, center_y = size // 2, size // 2
            
            if piece_type == 'pawn':
                pygame.draw.circle(surface, piece_color, (center_x, center_y - 10), 12)
                pygame.draw.circle(surface, outline_color, (center_x, center_y - 10), 12, 2)
                pygame.draw.rect(surface, piece_color, (center_x - 15, center_y + 5, 30, 20))
                pygame.draw.rect(surface, outline_color, (center_x - 15, center_y + 5, 30, 20), 2)
                
            elif piece_type == 'knight':
                points = [(center_x - 10, center_y + 20), (center_x - 15, center_y), 
                         (center_x - 5, center_y - 15), (center_x + 10, center_y - 10),
                         (center_x + 15, center_y + 5), (center_x + 10, center_y + 20)]
                pygame.draw.polygon(surface, piece_color, points)
                pygame.draw.polygon(surface, outline_color, points, 2)
                
            elif piece_type == 'bishop':
                pygame.draw.circle(surface, piece_color, (center_x, center_y - 15), 8)
                pygame.draw.circle(surface, outline_color, (center_x, center_y - 15), 8, 2)
                points = [(center_x, center_y - 5), (center_x - 15, center_y + 20),
                         (center_x + 15, center_y + 20)]
                pygame.draw.polygon(surface, piece_color, points)
                pygame.draw.polygon(surface, outline_color, points, 2)
                
            elif piece_type == 'rook':
                pygame.draw.rect(surface, piece_color, (center_x - 12, center_y - 15, 24, 35))
                pygame.draw.rect(surface, outline_color, (center_x - 12, center_y - 15, 24, 35), 2)
                for i in range(3):
                    x = center_x - 10 + i * 10
                    pygame.draw.rect(surface, piece_color, (x, center_y - 20, 6, 5))
                    pygame.draw.rect(surface, outline_color, (x, center_y - 20, 6, 5), 2)
                
            elif piece_type == 'queen':
                pygame.draw.circle(surface, piece_color, (center_x, center_y + 5), 18)
                pygame.draw.circle(surface, outline_color, (center_x, center_y + 5), 18, 2)
                for i in range(5):
                    angle = -90 + i * 45
                    x = center_x + int(20 * pygame.math.Vector2(1, 0).rotate(angle).x)
                    y = center_y - 5 + int(20 * pygame.math.Vector2(1, 0).rotate(angle).y)
                    pygame.draw.circle(surface, piece_color, (x, y), 5)
                    pygame.draw.circle(surface, outline_color, (x, y), 5, 2)
                
            elif piece_type == 'king':
                pygame.draw.circle(surface, piece_color, (center_x, center_y + 5), 18)
                pygame.draw.circle(surface, outline_color, (center_x, center_y + 5), 18, 2)
                pygame.draw.line(surface, outline_color, (center_x, center_y - 20), 
                               (center_x, center_y - 10), 3)
                pygame.draw.line(surface, outline_color, (center_x - 5, center_y - 15), 
                               (center_x + 5, center_y - 15), 3)
            
            pieces[symbol] = surface
        
        return pieces
        
    def draw_board(self):
        """Draw the chess board"""
        for row in range(8):
            for col in range(8):
                color = self.WHITE_SQUARE if (row + col) % 2 == 0 else self.BLACK_SQUARE
                pygame.draw.rect(self.screen, color,
                               (col * self.SQUARE_SIZE, row * self.SQUARE_SIZE,
                                self.SQUARE_SIZE, self.SQUARE_SIZE))
                
                if col == 0:
                    label = self.font_small.render(str(8 - row), True, (100, 100, 100))
                    self.screen.blit(label, (5, row * self.SQUARE_SIZE + 5))
                if row == 7:
                    label = self.font_small.render(chr(97 + col), True, (100, 100, 100))
                    self.screen.blit(label, (col * self.SQUARE_SIZE + self.SQUARE_SIZE - 20,
                                            self.BOARD_SIZE - 25))
    
    def draw_pieces(self):
        """Draw the chess pieces"""
        for row in range(8):
            for col in range(8):
                square = chess.square(col, 7 - row)
                piece = self.board.piece_at(square)
                
                if piece:
                    piece_image = self.piece_images.get(piece.symbol())
                    if piece_image:
                        x = col * self.SQUARE_SIZE + 5
                        y = row * self.SQUARE_SIZE + 5
                        self.screen.blit(piece_image, (x, y))
    
    def draw_selected(self):
        """Highlight selected square"""
        if self.selected_square is not None:
            col = chess.square_file(self.selected_square)
            row = 7 - chess.square_rank(self.selected_square)
            
            s = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE))
            s.set_alpha(128)
            s.fill((50, 205, 50))
            self.screen.blit(s, (col * self.SQUARE_SIZE, row * self.SQUARE_SIZE))
    
    def draw_info(self):
        """Draw info panel with evaluation"""
        pygame.draw.rect(self.screen, self.BG_COLOR,
                        (0, self.BOARD_SIZE, self.WIDTH, self.INFO_HEIGHT))
        
        # Status message
        status_color = (46, 204, 113) if not self.ai_thinking else (243, 156, 18)
        
        # Add check/checkmate warning to message
        display_message = self.message
        if self.board.is_check() and not self.board.is_game_over():
            display_message += " - CHECK!"
            status_color = (231, 76, 60)  # Red for check
        
        status_text = self.font_medium.render(display_message, True, status_color)
        self.screen.blit(status_text, (10, self.BOARD_SIZE + 10))
        
        # Position evaluation (if available)
        if hasattr(self.ai, 'last_eval'):
            eval_text = f"Eval: {self.ai.last_eval:+.1f}"
            eval_color = (46, 204, 113) if self.ai.last_eval < 0 else (231, 76, 60)
            if abs(self.ai.last_eval) < 50:
                eval_color = (241, 196, 15)
            eval_surface = self.font_small.render(eval_text, True, eval_color)
            self.screen.blit(eval_surface, (10, self.BOARD_SIZE + 45))
        
        # Move count
        move_text = self.font_small.render(f"Moves: {len(self.board.move_stack)}",
                                          True, self.TEXT_COLOR)
        self.screen.blit(move_text, (10, self.BOARD_SIZE + 75))
        
        # Instructions
        inst_text = self.font_tiny.render("Click to move | R: Reset | ESC: Quit",
                                          True, (200, 200, 200))
        self.screen.blit(inst_text, (10, self.BOARD_SIZE + 105))
    
    def get_square_from_mouse(self, pos):
        """Convert mouse position to chess square"""
        x, y = pos
        if y > self.BOARD_SIZE:
            return None
        col = x // self.SQUARE_SIZE
        row = y // self.SQUARE_SIZE
        return chess.square(col, 7 - row)
    
    def handle_click(self, pos):
        """Handle mouse click"""
        # Don't allow moves if game is over
        if self.board.is_game_over():
            return
        
        # Don't allow moves during AI turn
        if self.board.turn != self.player_color or self.ai_thinking:
            return
        
        square = self.get_square_from_mouse(pos)
        if square is None:
            return
        
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.player_color:
                self.selected_square = square
        else:
            move = chess.Move(self.selected_square, square)
            
            # Check for promotion
            piece = self.board.piece_at(self.selected_square)
            if piece and piece.piece_type == chess.PAWN:
                if (piece.color == chess.WHITE and chess.square_rank(square) == 7) or \
                   (piece.color == chess.BLACK and chess.square_rank(square) == 0):
                    move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)
            
            if move in self.board.legal_moves:
                self.board.push(move)
                self.selected_square = None
                
                # Check if game ended with player's move
                if self.board.is_game_over():
                    self.check_game_over()
                else:
                    self.message = "AI is thinking..."
                    self.ai_thinking = True
            else:
                self.selected_square = None
    
    def ai_move(self):
        """AI makes a move"""
        if not self.ai_thinking:
            return
        
        # Check if game is already over
        if self.board.is_game_over():
            self.ai_thinking = False
            self.check_game_over()
            return
        
        # Make AI move
        move = self.ai.get_best_move(self.board)
        self.board.push(move)
        self.ai_thinking = False
        
        # Check game state after AI move
        if self.board.is_game_over():
            self.check_game_over()
        else:
            self.message = "Your turn"
    
    def check_game_over(self):
        """Check if game is over"""
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
            self.message = f"Checkmate! {winner} wins!"
        elif self.board.is_stalemate():
            self.message = "Stalemate - Draw!"
        elif self.board.is_insufficient_material():
            self.message = "Draw - Insufficient material"
        elif self.board.is_fifty_moves():
            self.message = "Draw - Fifty move rule"
        elif self.board.is_repetition():
            self.message = "Draw - Repetition"
        else:
            self.message = "Game Over"
    
    def reset_game(self):
        """Reset the game"""
        self.board = chess.Board()
        self.selected_square = None
        self.message = "Your turn (White)"
        self.ai_thinking = False
    
    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        ai_move_timer = 0
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    if event.key == pygame.K_r:
                        self.reset_game()
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(pygame.mouse.get_pos())
            
            # AI move with delay
            if self.ai_thinking:
                ai_move_timer += clock.get_time()
                if ai_move_timer > 300:  # 300ms delay
                    self.ai_move()
                    ai_move_timer = 0
            else:
                ai_move_timer = 0  # Reset timer when not AI's turn
            
            # Always check game over status
            if self.board.is_game_over() and self.message not in ["Checkmate! White wins!", "Checkmate! Black wins!", "Stalemate - Draw!", "Draw - Insufficient material", "Draw - Fifty move rule", "Draw - Repetition"]:
                self.check_game_over()
            
            # Draw everything
            self.draw_board()
            self.draw_selected()
            self.draw_pieces()
            self.draw_info()
            
            pygame.display.flip()
            clock.tick(60)

if __name__ == "__main__":
    try:
        print("="*70)
        print("ADVANCED CHESS AI - VISUAL GAME")
        print("="*70)
        print("\nFeatures:")
        print("✓ Multi-task trained model (moves + evaluation)")
        print("✓ Enhanced 791-feature input")
        print("✓ Real-time position evaluation display")
        print("✓ Trained on Stockfish data")
        print("\n" + "="*70 + "\n")
        
        game = ChessGUI()
        print("Starting game...\n")
        game.run()
    except FileNotFoundError:
        print("\n Error: chess_model_advanced.pth not found!")
        print("\nPlease train the model first by running:")
        print("  python train_advanced_model.py")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have installed:")
        print("  pip install pygame chess torch numpy")