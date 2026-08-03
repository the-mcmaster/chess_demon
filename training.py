"""
training.py

Self-play data generation and unsupervised training loop.

One epoch:
    - play games_per_epoch self-play games (full games, model_white plays
      the white side, model_black plays the black side - the same model
      instance can be passed for both)
    - a game ends automatically after 200 full moves (400 plies) if it
      hasn't already ended naturally
    - every position/move played in every game is recorded, tagged with
      which side made that move
    - after all games, each recorded position is labeled with the average
      outcome seen for that exact position (from the mover's perspective:
      1.0 win, -1.0 loss, 0.0 draw, per the outcome dict below)
    - each model's value head is trained to regress (via tanh output, so
      targets live in [-1, 1]) toward that average
"""

import random
import os
import sys

import chess
import chess.pgn
import torch
import torch.nn.functional as F

from model import ChessTransformer, board_to_tensor

MAX_FULL_MOVES = 200  # 200 full moves = 400 plies, per spec

# --- 73-plane move encoding -------------------------------------------------
# 0-55  : 8 compass directions x 7 distances (sliding queen-like moves,
#         including king steps, castling, and pawn pushes/captures/queen
#         promotions - anything that moves in a straight or diagonal line)
# 56-63 : 8 knight-move offsets
# 64-72 : 9 underpromotions (3 directions x 3 piece choices: N, B, R)

DIRECTIONS = [
    (1, 0),    # N
    (1, 1),    # NE
    (0, 1),    # E
    (-1, 1),   # SE
    (-1, 0),   # S
    (-1, -1),  # SW
    (0, -1),   # W
    (1, -1),   # NW
]

KNIGHT_OFFSETS = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1),
]

UNDERPROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def _canonical_rc(square: chess.Square, mover: bool):
    """Row/col of a square in the mover-relative frame that model.py's
    board_to_tensor uses (rank flipped whenever black is to move, file
    unchanged)."""
    row = chess.square_rank(square)
    col = chess.square_file(square)
    if mover == chess.BLACK:
        row = 7 - row
    return row, col


def move_to_plane_index(move: chess.Move, board: chess.Board) -> int:
    """
    Map a legal chess.Move to its plane index (0-72) in the policy tensor.

    The policy tensor is produced from the mover-relative board encoding
    (see model.py), so the move's from/to squares are first converted into
    that same canonical frame before computing direction/distance - the
    board's real, un-flipped coordinates are only used for pushing the
    move onto the board itself.
    """
    mover = board.turn
    from_row, from_col = _canonical_rc(move.from_square, mover)
    to_row, to_col = _canonical_rc(move.to_square, mover)
    dr = to_row - from_row
    dc = to_col - from_col

    if move.promotion is not None and move.promotion != chess.QUEEN:
        # dc is -1/0/1 for the mover's capture-left / push / capture-right
        dir_idx = dc + 1
        piece_idx = UNDERPROMOTION_PIECES.index(move.promotion)
        return 64 + dir_idx * 3 + piece_idx

    if (abs(dr), abs(dc)) in ((2, 1), (1, 2)):
        return 56 + KNIGHT_OFFSETS.index((dr, dc))

    # sliding move: normalize to a unit direction + distance (1-7)
    dr_sign = (dr > 0) - (dr < 0)
    dc_sign = (dc > 0) - (dc < 0)
    distance = max(abs(dr), abs(dc))
    dir_idx = DIRECTIONS.index((dr_sign, dc_sign))
    return dir_idx * 7 + (distance - 1)


def select_move_from_output(
    policy_logits: torch.Tensor,
    board: chess.Board,
    temperature: float = 1.0,
) -> chess.Move:
    """
    Turn the model's 8x8x73 policy tensor into a move.

    For every legal move, look up its (from_square, plane) entry in
    policy_logits to get a single scalar value, then softmax those values
    (with temperature) into a probability distribution over legal moves
    and sample from it.

    temperature > 1.0 flattens the distribution (more exploration),
    temperature < 1.0 sharpens it (more exploitation), and
    temperature -> 0 behaves like argmax (always the highest-valued move).
    """
    legal_moves = list(board.legal_moves)
    mover = board.turn

    move_values = []
    for move in legal_moves:
        from_row, from_col = _canonical_rc(move.from_square, mover)
        plane_idx = move_to_plane_index(move, board)
        move_values.append(policy_logits[from_row, from_col, plane_idx])

    move_values = torch.stack(move_values)

    if temperature <= 0:
        best_idx = torch.argmax(move_values).item()
        return legal_moves[best_idx]

    probs = F.softmax(move_values / temperature, dim=0)
    chosen_idx = torch.multinomial(probs, 1).item()
    return legal_moves[chosen_idx]


def play_game(
    model_white: ChessTransformer,
    model_black: ChessTransformer,
    device: str = "cpu",
    temperature: float = 1.0,
    **kwargs
):
    """
    Play one full self-play game.

    Returns:
        history: list of (input_tensor, mover_color) for every ply played
        outcome: dict mapping chess.WHITE / chess.BLACK -> result in {-1.0, 0.0, 1.0}
                    - 0.0 for both sides if the game results in a draw
    """
    attempt = 0
    while True:
        attempt += 1
        board = chess.Board()
        history = []

        full_move_count = 0
        while not board.is_game_over(claim_draw=True) and full_move_count < MAX_FULL_MOVES:
            input_tensor = board_to_tensor(board).unsqueeze(0).to(device)

            model = model_white if board.turn == chess.WHITE else model_black
            with torch.no_grad():
                policy_logits, _value = model(input_tensor)

            move = select_move_from_output(policy_logits[0], board, temperature=temperature)

            history.append((input_tensor.squeeze(0).cpu(), board.turn))
            board.push(move)

            if board.turn == chess.WHITE:
                # a full move just completed (black just moved)
                full_move_count += 1

        game = chess.pgn.Game.from_board(board)

        game.headers["Event"] = f"Epoch {epoch} Game {kwargs['Count']} Attempt {attempt}"
        game.headers["White"] = kwargs["White"]
        game.headers["Black"] = kwargs["Black"]
        game.headers["Result"] = board.result()

        os.makedirs(f"{kwargs['Epoch']}", exist_ok=True)
        with open(f"{kwargs['Epoch']}/epoch{kwargs['Epoch']}game{kwargs['Count']}attempt{attempt}.pgn", "w", encoding="utf-8") as pgn_file:
            pgn_file.write(str(game))

        if board.is_game_over(claim_draw=True):
            result = board.result(claim_draw=True)
        else:
            # hit the 200-full-move cap without a natural conclusion
            continue

        break

    if result == "1-0":
        outcome = {chess.WHITE: 1.0, chess.BLACK: -1.0}
    elif result == "0-1":
        outcome = {chess.WHITE: -1.0, chess.BLACK: 1.0}
    else:
        outcome = {chess.WHITE: 0.0, chess.BLACK: 0.0}

    return history, outcome, attempt


def play_epoch(
    model_white: ChessTransformer,
    model_black: ChessTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: str = "cpu",
    games_per_epoch: int = 1024,
    train_batch_size: int = 256,
    temperature: float = 1.0,
):
    """
    Play games_per_epoch self-play games, then train each model's value
    head on the resulting (position, average outcome) pairs.
    """
    # keyed by the raw bytes of the position tensor, since torch.Tensor
    # isn't hashable and can't be used as a dict key directly
    all_position_results = {chess.WHITE: {}, chess.BLACK: {}}

    summary = {'white': 0, 'black': 0, 'draw': 0, 'attempts': 0, 'states': 0}
    for game_idx in range(games_per_epoch):
        history, outcome, attempts = play_game(model_white, model_black, device=device, temperature=temperature, Epoch=epoch, White=f"Generation {model_white.epoch}", Black=f"Generation {model_black.epoch}", Count=game_idx + 1)
        for position_tensor, mover_color in history:
            key = position_tensor.numpy().tobytes()
            entry = all_position_results[mover_color].setdefault(
                key, {"tensor": position_tensor, "results": []}
            )
            entry["results"].append(outcome[mover_color])
        
        if outcome[chess.WHITE] == 1: summary['white'] += 1
        elif outcome[chess.WHITE] == -1: summary['black'] += 1
        else: summary['draw'] += 1
        summary['attempts'] += attempts
        summary['states'] += len(all_position_results[chess.WHITE].keys())
        summary['states'] += len(all_position_results[chess.BLACK].keys())

    average_loss_white_model = eval_positions(
        model_white, all_position_results[chess.WHITE], device=device, train_batch_size=train_batch_size
    )
    average_loss_black_model = eval_positions(
        model_black, all_position_results[chess.BLACK], device=device, train_batch_size=train_batch_size
    )

    return (average_loss_white_model, average_loss_black_model, summary)


def eval_positions(model, position_entries, device: str = "cpu", train_batch_size: int = 256):
    model.eval()

    all_tensors = []
    all_targets = []
    for entry in position_entries.values():
        all_tensors.append(entry["tensor"])
        results = entry["results"]
        all_targets.append(sum(results) / len(results))

    data = torch.stack(all_tensors)                            # (N, 19, 8, 8)
    targets = torch.tensor(all_targets, dtype=torch.float32)   # (N,)

    model.train()
    total_loss = 0.0
    num_batches = 0

    perm = torch.randperm(data.shape[0])
    for start in range(0, data.shape[0], train_batch_size):
        idx = perm[start:start + train_batch_size]
        batch_x = data[idx].to(device)
        batch_y = targets[idx].to(device)

        optimizer.zero_grad()
        _policy_logits, value_preds = model(batch_x)
        loss = F.mse_loss(value_preds, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device `{device}`")
    model = ChessTransformer(0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    start_epoch = None
    if len(sys.argv) >= 2:
        start_epoch = int(sys.argv[1])
        print(f"Attempting to load at Epoch {start_epoch}")
        model.load_state_dict(torch.load(f"model{start_epoch}.pth", weights_only=True))
        model.epoch = start_epoch
        print(f"Successfully loaded Epoch {start_epoch}")
    else:
        print("No model number provided. Starting at Epoch 0")


    def print_summary(summary):
        print(f"\tEpoch Game Summary")
        print(f"\tAttempts:     {summary['attempts']}")
        print(f"\tBoard States: {summary['states']}")
        print(f"\tWhite Wins:   {summary['white']}")
        print(f"\tBlack Wins:   {summary['black']}")
        print(f"\tDraws:        {summary['draw']}")

    for epoch in range(0 if not start_epoch else start_epoch + 1, 20):
        model.epoch = epoch

        print()
        print(f"Beginning Epoch {epoch}")
        
        games_this_epoch = min(1024, 64 * (2**epoch))
        avg_loss_white, avg_loss_black, summary = play_epoch(
            model, model, optimizer, epoch, device=device, games_per_epoch=games_this_epoch, train_batch_size=games_this_epoch // 4
        )
        
        print(f"Epoch {epoch} complete. Average value loss white: {avg_loss_white:.4f} Average value loss black: {avg_loss_black:.4f}")
        print_summary(summary)
        print(f"Saving Epoch Model as `model{epoch}.pth`")
        torch.save(model.state_dict(), f"model{epoch}.pth")