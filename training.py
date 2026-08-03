"""
training.py

Self-play data generation and unsupervised training loop.

One epoch:
    - play 1000 self-play games (full games, one model playing both sides)
    - a game ends automatically after 200 full moves (400 plies) if it
      hasn't already ended naturally
    - every position/move played in every game is recorded, tagged with
      which side made that move
    - after all games, each recorded position is labeled with its game's
      outcome from the mover's perspective (1.0 win, 0.5 draw, 0.0 loss)
    - the model is trained (value head) to regress toward that outcome

Note on "500 games as white / 500 as black": since a single self-play game
already records moves for both colors, this split doesn't change what data
a given game produces. If you intended something more specific (e.g.
tracking win-rate by color, or two different opening setups), let me know
and I'll adjust play_epoch accordingly - for now it just plays 1000
self-play games total.
"""

import random

import chess
import torch
import torch.nn.functional as F

from model import ChessTransformer, board_to_tensor

MAX_FULL_MOVES = 200  # 200 full moves = 400 plies, per spec

def select_move_from_output(policy_logits: torch.Tensor, board: chess.Board) -> chess.Move:
    """
    DUMMY PLACEHOLDER.

    Should translate the model's 8x8x73 policy tensor for the current
    board into a legal chess.Move:
      1. Map the policy tensor's (from_square, plane) indices to candidate
         moves using your 73-plane move encoding (56 sliding + 8 knight +
         9 underpromotion).
      2. Mask out anything not in board.legal_moves.
      3. Select a move (argmax for greedy play, or sample for exploration).

    For now this just returns a random legal move so the rest of the
    pipeline (self-play, data collection, training) can run end-to-end.
    """
    legal_moves = list(board.legal_moves)

    return random.choice(legal_moves)


def play_game(model_white: ChessTransformer, model_black: ChessTransformer, device: str = "cpu"):
    """
    Play one full self-play game.

    Returns:
        history: list of (input_tensor, mover_color) for every ply played
        outcome: dict mapping chess.WHITE / chess.BLACK -> result in {0.0, 1.0}
                    - 0.0 for both sides if game results in a draw
    """
    board = chess.Board()
    history = []

    full_move_count = 0
    while not board.is_game_over(claim_draw=True) and full_move_count < MAX_FULL_MOVES:
        input_tensor = board_to_tensor(board).unsqueeze(0).to(device)

        model = model_white if board.turn == chess.WHITE else model_black
        with torch.no_grad():
            policy_logits, _value = model(input_tensor)

        move = select_move_from_output(policy_logits[0], board)

        history.append((input_tensor.squeeze(0).cpu(), board.turn))
        board.push(move)

        if board.turn == chess.WHITE:
            # a full move just completed (black just moved)
            full_move_count += 1

    if board.is_game_over(claim_draw=True):
        result = board.result(claim_draw=True)
    else:
        # hit the 200-full-move cap without a natural conclusion
        result = "1/2-1/2"

    if result == "1-0":
        outcome = {chess.WHITE: 1.0, chess.BLACK: 0.0}
    elif result == "0-1":
        outcome = {chess.WHITE: 0.0, chess.BLACK: 1.0}
    else:
        outcome = {chess.WHITE: 0.0, chess.BLACK: 0.0}

    return history, outcome


def play_epoch(
    model_white: ChessTransformer,
    model_black: ChessTransformer,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    games_per_epoch: int = 1024,
    train_batch_size: int = 256,
):
    """
    Play games_per_epoch self-play games, then train the model's value
    head on the resulting (position, outcome) pairs.
    """
    
    all_position_results = {chess.WHITE: {}, chess.BLACK: {}}

    for game_idx in range(games_per_epoch):
        history, outcome = play_game(model_white, model_black, device=device)
        for position_tensor, mover_color in history:
            results = all_position_results[mover_color].setdefault(position_tensor, [])
            results.append(outcome[mover_color])

    average_loss_white_model = eval_positions(model_white, all_position_results[chess.WHITE],train_batch_size=train_batch_size)
    average_loss_black_model = eval_positions(model_black, all_position_results[chess.BLACK],train_batch_size=train_batch_size)

    return (average_loss_white_model, average_loss_black_model, )

def eval_positions(model, all_position_results, train_batch_size: int = 256,):
    model.eval()

    all_tensors = []
    all_targets = []
    for position, results in all_position_results.items():
        all_tensors.append(position)
        average = torch.tensor(results).mean()
        all_targets.append(average)

    data = torch.stack(all_tensors)               # (N, 19, 8, 8)
    targets = torch.tensor(all_targets, dtype=torch.float32)  # (N, 2)

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
    model = ChessTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        avg_loss_white, avg_loss_black = play_epoch(model, model, optimizer, device=device, games_per_epoch=10, train_batch_size=10)
        print(f"Epoch {epoch} complete. Average value loss white: {avg_loss_white:.4f} Average value loss black: {avg_loss_black:.4f}")