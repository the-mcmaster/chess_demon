"""
model.py

Board <-> tensor encoding and the transformer-based chess model.

The board is encoded from the MOVER'S perspective (canonicalized), not in
fixed white/black terms:
    - "my" pieces always occupy channels 0-5, "opponent" pieces channels 6-11,
      regardless of which color is actually moving.
    - the rank axis is flipped (row -> 7 - row) whenever black is to move,
      so the mover's pieces always start near row 0 and advance toward
      row 7 - the same geometric picture the model sees whether it's
      playing white or black. The file axis is never flipped (a
      kingside/queenside-preserving mirror, not a full board rotation).
    - castling rights are similarly relabeled as "mover" / "opponent"
      instead of fixed white/black.

Without this, the exact same piece arrangement would look identical to the
network whether it's white's move or black's, even though the correct
value (win/loss/draw) and correct pawn-push direction depend entirely on
whose move it is.

Input encoding (per board):  8 x 8 x 19
    channels 0-5   : mover's pieces (pawn, knight, bishop, rook, queen, king)
    channels 6-11  : opponent's pieces (same piece order)
    channel 12     : en passant target square (one-hot, in canonical coords)
    channels 13-16 : castling rights (mover-K, mover-Q, opponent-K, opponent-Q)
    channels 17-18 : "special" castling rules (placeholders - see TODO below)

The model concatenates 2 positional-encoding channels (row, col normalized
to 0-1, in canonical coordinates) internally, giving the 21-length
per-square vector used as the transformer token embedding (64 tokens, one
per square).

Output: policy tensor of shape 8 x 8 x 73 (also in canonical coordinates -
see move_to_plane_index / select_move_from_output in training.py for how
legal moves are looked up in this frame), plus a scalar value in [-1, 1]
representing the mover's expected outcome: -1.0 loss, 0.0 draw, 1.0 win.

Policy channel layout (73 total), matching the move-encoding scheme you
described:
    0-55  : 8 directions x 7 distances (sliding moves for Q/R/B/K/pawn pushes)
    56-63 : 8 knight-move offsets
    64-72 : 9 underpromotion options (3 directions x 3 piece choices: N, B, R)
"""

import chess
import torch
import torch.nn as nn

# Ordering used for the 6 "piece type" channels within each mover/opponent block.
PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]

NUM_INPUT_CHANNELS = 19  # 12 piece planes + 1 en passant + 6 castling
NUM_POSITIONAL_CHANNELS = 2
D_MODEL = NUM_INPUT_CHANNELS + NUM_POSITIONAL_CHANNELS  # 21, per-square token length
NUM_POLICY_CHANNELS = 73


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Convert a python-chess Board into the 19 x 8 x 8 input tensor, from the
    perspective of the side to move (board.turn). See module docstring for
    the canonicalization details.
    """
    mover = board.turn
    planes = torch.zeros(NUM_INPUT_CHANNELS, 8, 8, dtype=torch.float32)

    # Channels 0-11: piece planes, mover's pieces first.
    for square, piece in board.piece_map().items():
        row = chess.square_rank(square)   # 0 (rank 1) .. 7 (rank 8)
        col = chess.square_file(square)   # 0 (file a) .. 7 (file h)
        if mover == chess.BLACK:
            row = 7 - row
        piece_idx = PIECE_TYPES.index(piece.piece_type)
        color_offset = 0 if piece.color == mover else 6
        planes[piece_idx + color_offset, row, col] = 1.0

    # Channel 12: en passant target square (canonical coordinates).
    if board.ep_square is not None:
        row = chess.square_rank(board.ep_square)
        col = chess.square_file(board.ep_square)
        if mover == chess.BLACK:
            row = 7 - row
        planes[12, row, col] = 1.0

    # Channels 13-16: castling rights, relabeled mover/opponent instead of
    # fixed white/black, broadcast across the whole plane (a common
    # convention so the transformer sees the flag at every square).
    opponent = not mover
    planes[13, :, :] = 1.0 if board.has_kingside_castling_rights(mover) else 0.0
    planes[14, :, :] = 1.0 if board.has_queenside_castling_rights(mover) else 0.0
    planes[15, :, :] = 1.0 if board.has_kingside_castling_rights(opponent) else 0.0
    planes[16, :, :] = 1.0 if board.has_queenside_castling_rights(opponent) else 0.0

    # Channels 17-18: special castling rules.
    # TODO: define and populate these for your 2 special castling moves.
    # Left at zero for now.
    planes[17, :, :] = 0.0
    planes[18, :, :] = 0.0

    return planes


class ChessTransformer(nn.Module):
    def __init__(
        self,
        epoch: int,
        d_model: int = D_MODEL,
        nhead: int = 3,
        num_layers: int = 12,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.epoch = epoch
        self.d_model = d_model

        # Positional encoding: 2 channels holding the (row, col) coordinates
        # of each square, normalized to [0, 1], in canonical coordinates.
        coords = torch.zeros(NUM_POSITIONAL_CHANNELS, 8, 8)
        for row in range(8):
            for col in range(8):
                coords[0, row, col] = row / 7.0
                coords[1, row, col] = col / 7.0
        self.register_buffer("positional_encoding", coords)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.policy_head = nn.Linear(d_model, NUM_POLICY_CHANNELS)

        self.value_head = nn.Sequential(
            nn.Linear(d_model * 64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        """
        x: (batch, 19, 8, 8), in canonical (mover-relative) coordinates.
        returns:
            policy_logits: (batch, 8, 8, 73), in canonical coordinates
            value: (batch,) in [-1, 1] - the mover's expected outcome
                (-1 loss, 0 draw, 1 win)
        """
        batch = x.shape[0]

        pos = self.positional_encoding.unsqueeze(0).expand(batch, -1, -1, -1)
        x = torch.cat([x, pos], dim=1)  # (batch, 21, 8, 8)

        # Flatten the 8x8 board into 64 tokens, row-major (row * 8 + col),
        # matching board_to_tensor's [channel, row, col] layout.
        tokens = x.flatten(2).permute(0, 2, 1)  # (batch, 64, 21)

        encoded = self.encoder(tokens)  # (batch, 64, 21)

        policy_logits = self.policy_head(encoded)          # (batch, 64, 73)
        policy_logits = policy_logits.view(batch, 8, 8, NUM_POLICY_CHANNELS)

        encoded_flat = encoded.flatten(1)  # (batch, 64 * 21)
        value = self.value_head(encoded_flat).squeeze(-1)  # (batch,)

        return policy_logits, value