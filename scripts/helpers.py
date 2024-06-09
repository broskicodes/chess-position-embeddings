import random
import chess.pgn
import chess
import torch
from torch.utils.data import Dataset, DataLoader


PIECE_TO_INDEX = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}

class PositionDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx]

def pos_to_tensor(fen, device):
    parts = fen.split(" ")
    wtm = parts[1] == "w"
    castling_rights = parts[2]

    board = chess.Board(fen)
    tensor = torch.zeros(15, 8, 8, device=device)

    for row in range(8): 
        for col in range(8):
            sqr = chess.square(col, 7 - row)
            piece = board.piece_at(sqr)
            if piece != None:
                p = piece.symbol()
                idx = PIECE_TO_INDEX[p]
                tensor[idx, row, col] = 1 if p.isupper() else -1
                  
    # Encode castling rights
    if 'K' in castling_rights:
        tensor[12, 0, 0] = 1
    if 'Q' in castling_rights:
        tensor[12, 0, 7] = 1
    if 'k' in castling_rights:
        tensor[13, 7, 0] = -1  
    if 'q' in castling_rights:
        tensor[13, 7, 7] = -1
          
    # Encode side to move
    tensor[14] = 1 if wtm else -1
      
    return tensor

def tensor_to_pos(tensor):
    board = chess.Board(None)
    piece_symbols = list(PIECE_TO_INDEX.keys())
    
    # Decode the board pieces
    for idx, piece_symbol in enumerate(piece_symbols[:12]):
        mask = tensor[idx].abs() > 0
        positions = mask.nonzero(as_tuple=True)
        for row, col in zip(*positions):
            square = chess.square(col, 7 - row)
            board.set_piece_at(square, chess.Piece.from_symbol(piece_symbol))
    
    # Decode castling rights
    castling_rights = ''
    if tensor[12, 0, 0] == 1:
        castling_rights += 'K'
    if tensor[12, 0, 7] == 1:
        castling_rights += 'Q'
    if tensor[13, 7, 0] == -1:
        castling_rights += 'k'
    if tensor[13, 7, 7] == -1:
        castling_rights += 'q'
    board.set_castling_fen(castling_rights)
    
    # Decode side to move
    side_to_move = 'w' if tensor[14].mean() > 0 else 'b'
    board.turn = True if side_to_move == 'w' else False
    
    return board.fen()
    

def get_datasets(positions, device):
    tensors = [pos_to_tensor(pos, device) for pos in positions]
    random.shuffle(tensors)

    # Calculate the indices for splitting
    total_tensors = len(tensors)
    train_end = int(total_tensors * 0.8)
    val_end = int(total_tensors * 0.9)

    # Split the tensors into train, validation, and test sets
    train_tensors = tensors[:train_end]
    val_tensors = tensors[train_end:val_end]
    test_tensors = tensors[val_end:]

    # Create datasets for each split
    train_dataset = PositionDataset(train_tensors)
    val_dataset = PositionDataset(val_tensors)
    test_dataset = PositionDataset(test_tensors)

    return train_dataset, val_dataset, test_dataset
