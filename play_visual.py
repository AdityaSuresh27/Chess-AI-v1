import chess
import torch
import torch.nn as nn
import numpy as np
import pygame
import sys

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

class ChessAI:
    def __init__(self, model_path='chess_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ChessNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
    
    def board_to_feature_vector(self, board):
        features = []
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
        features.append(1 if board.turn == chess.WHITE else 0)
        features.append(1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
        features.append(1 if board.has_queenside_castling_rights(chess.WHITE) else 0)
        features.append(1 if board.has_kingside_castling_rights(chess.BLACK) else 0)
        features.append(1 if board.has_queenside_castling_rights(chess.BLACK) else 0)
        
        return np.array(features, dtype=np.float32)
    
    def index_to_move(self, index):
        from_square = index // 64
        to_square = index % 64
        return chess.Move(from_square, to_square)
    
    def get_best_move(self, board, top_k=100):
        features = self.board_to_feature_vector(board)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(features_tensor)
            probabilities = torch.softmax(output, dim=1)
            top_moves = torch.topk(probabilities, top_k)
        
        legal_moves = list(board.legal_moves)
        
        for idx in top_moves.indices[0]:
            move = self.index_to_move(idx.item())
            for legal_move in legal_moves:
                if legal_move.from_square == move.from_square and legal_move.to_square == move.to_square:
                    return legal_move
        
        import random
        return random.choice(legal_moves)

class ChessGUI:
    def __init__(self):
        pygame.init()
        
        # Constants
        self.SQUARE_SIZE = 80
        self.BOARD_SIZE = 8 * self.SQUARE_SIZE
        self.INFO_HEIGHT = 150
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
        pygame.display.set_caption("Chess AI - Visual Game")
        
        self.board = chess.Board()
        self.ai = ChessAI()
        self.selected_square = None
        self.player_color = chess.WHITE
        
        # Fonts
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        
        # Create piece images
        self.piece_images = self.create_piece_images()
        
        self.message = "Your turn (White)"
        self.ai_thinking = False
    
    def create_piece_images(self):
        """Create simple graphical representations of chess pieces"""
        pieces = {}
        size = self.SQUARE_SIZE - 10
        
        # Define piece shapes using circles and polygons
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
                # Circle for head
                pygame.draw.circle(surface, piece_color, (center_x, center_y - 10), 12)
                pygame.draw.circle(surface, outline_color, (center_x, center_y - 10), 12, 2)
                # Base
                pygame.draw.rect(surface, piece_color, (center_x - 15, center_y + 5, 30, 20))
                pygame.draw.rect(surface, outline_color, (center_x - 15, center_y + 5, 30, 20), 2)
                
            elif piece_type == 'knight':
                # Horse shape
                points = [(center_x - 10, center_y + 20), (center_x - 15, center_y), 
                         (center_x - 5, center_y - 15), (center_x + 10, center_y - 10),
                         (center_x + 15, center_y + 5), (center_x + 10, center_y + 20)]
                pygame.draw.polygon(surface, piece_color, points)
                pygame.draw.polygon(surface, outline_color, points, 2)
                
            elif piece_type == 'bishop':
                # Triangle with circle on top
                pygame.draw.circle(surface, piece_color, (center_x, center_y - 15), 8)
                pygame.draw.circle(surface, outline_color, (center_x, center_y - 15), 8, 2)
                points = [(center_x, center_y - 5), (center_x - 15, center_y + 20),
                         (center_x + 15, center_y + 20)]
                pygame.draw.polygon(surface, piece_color, points)
                pygame.draw.polygon(surface, outline_color, points, 2)
                
            elif piece_type == 'rook':
                # Castle tower
                pygame.draw.rect(surface, piece_color, (center_x - 12, center_y - 15, 24, 35))
                pygame.draw.rect(surface, outline_color, (center_x - 12, center_y - 15, 24, 35), 2)
                # Battlements
                for i in range(3):
                    x = center_x - 10 + i * 10
                    pygame.draw.rect(surface, piece_color, (x, center_y - 20, 6, 5))
                    pygame.draw.rect(surface, outline_color, (x, center_y - 20, 6, 5), 2)
                
            elif piece_type == 'queen':
                # Crown with multiple points
                pygame.draw.circle(surface, piece_color, (center_x, center_y + 5), 18)
                pygame.draw.circle(surface, outline_color, (center_x, center_y + 5), 18, 2)
                for i in range(5):
                    angle = -90 + i * 45
                    x = center_x + int(20 * pygame.math.Vector2(1, 0).rotate(angle).x)
                    y = center_y - 5 + int(20 * pygame.math.Vector2(1, 0).rotate(angle).y)
                    pygame.draw.circle(surface, piece_color, (x, y), 5)
                    pygame.draw.circle(surface, outline_color, (x, y), 5, 2)
                
            elif piece_type == 'king':
                # Crown with cross
                pygame.draw.circle(surface, piece_color, (center_x, center_y + 5), 18)
                pygame.draw.circle(surface, outline_color, (center_x, center_y + 5), 18, 2)
                # Cross on top
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
                
                # Draw file/rank labels
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
                        # Center the piece in the square
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
        """Draw info panel"""
        pygame.draw.rect(self.screen, self.BG_COLOR,
                        (0, self.BOARD_SIZE, self.WIDTH, self.INFO_HEIGHT))
        
        # Status message
        status_color = (46, 204, 113) if not self.ai_thinking else (243, 156, 18)
        status_text = self.font_medium.render(self.message, True, status_color)
        self.screen.blit(status_text, (20, self.BOARD_SIZE + 20))
        
        # Instructions
        inst_text = self.font_small.render("Click squares to move | R: Reset | ESC: Quit",
                                          True, (200, 200, 200))
        self.screen.blit(inst_text, (20, self.BOARD_SIZE + 70))
        
        # Move count
        move_text = self.font_small.render(f"Moves: {len(self.board.move_stack)}",
                                          True, self.TEXT_COLOR)
        self.screen.blit(move_text, (20, self.BOARD_SIZE + 110))
    
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
        if self.board.is_game_over() or self.board.turn != self.player_color or self.ai_thinking:
            return
        
        square = self.get_square_from_mouse(pos)
        if square is None:
            return
        
        if self.selected_square is None:
            # Select a piece
            piece = self.board.piece_at(square)
            if piece and piece.color == self.player_color:
                self.selected_square = square
        else:
            # Try to move
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
                self.message = "AI is thinking..."
                self.ai_thinking = True
            else:
                self.selected_square = None
    
    def ai_move(self):
        """AI makes a move"""
        if not self.ai_thinking or self.board.is_game_over():
            return
        
        move = self.ai.get_best_move(self.board)
        self.board.push(move)
        self.ai_thinking = False
        
        if not self.board.is_game_over():
            self.message = "Your turn"
        else:
            self.check_game_over()
    
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
                if ai_move_timer > 500:  # 500ms delay
                    self.ai_move()
                    ai_move_timer = 0
            
            # Check game over
            if self.board.is_game_over() and not self.ai_thinking:
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
        print("Loading AI model...")
        game = ChessGUI()
        print("Starting game...")
        game.run()
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: This requires pygame. Install with:")
        print("pip install pygame")