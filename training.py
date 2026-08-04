"""
training.py

Self-play data generation and unsupervised training loop.

One epoch:
    - play games_per_epoch self-play games (full games, model_white plays
      the white side, model_black plays the black side - the same model
      instance can be passed for both)
    - a game ends automatically after 200 full moves (400 plies); if it
      hasn't naturally ended by then, the game is discarded and retried
      rather than counted (see play_game)
    - every position/move played in every game is recorded as its own
      occurrence: (board tensor, the move actually chosen there, and that
      game's final outcome for the mover). Unlike an earlier version of
      this file, occurrences are NOT deduplicated/averaged by position -
      the policy loss needs to know exactly which move was taken at each
      occurrence and what that specific game's outcome was, so collapsing
      repeat positions into one averaged target would throw away that
      pairing. (Training the value head on individual occurrences instead
      of pre-averaged targets is statistically equivalent in expectation -
      MSE over repeated samples converges to the same place as MSE over
      their pre-averaged mean - so this isn't a behavior regression for
      the value head, just a necessary simplification for the policy head.)
    - both heads are trained together, in one combined loss per batch:
        * value loss: MSE between the value head's prediction and the
          occurrence's actual game outcome (tanh output, so targets and
          predictions both live in [-1, 1])
        * policy loss: advantage-weighted negative log-likelihood of the
          move that was actually played, i.e. basic actor-critic /
          REINFORCE-with-baseline. The "advantage" is
          (actual outcome - value head's own prediction, detached) - how
          much better or worse the move did than the position was already
          expected to be worth. Moves that outperformed their baseline get
          reinforced (higher probability); moves that underperformed get
          discouraged.
      Both losses are summed and backpropagated in a single .backward()
      call, so the shared transformer trunk gets gradient from both
      objectives at once.
"""

import random
import os
import sys
import copy
import queue
import contextlib
import concurrent.futures

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


class ModelPool:
    """
    A pool of `pool_size` read-only replicas of a model, each bound to its
    own CUDA stream, checked out by threads during self-play and returned
    when a game finishes.

    Why a dedicated stream per replica, not just per replica weights:
    PyTorch funnels all CUDA kernel launches from every thread onto one
    shared default stream per device unless told otherwise. A stream is a
    FIFO queue of GPU work - kernels queued on the same stream still run
    one at a time no matter how many host threads enqueued them. Giving
    each replica its own stream lets the GPU scheduler genuinely overlap
    work from different threads, which multiple model instances alone do
    not provide (the weights are read-only during self-play, so multiple
    threads could safely share a single instance - the instances here
    exist to give each stream something to operate on, not to solve a
    correctness problem).

    On a non-CUDA device (device == "cpu"), streams aren't meaningful, so
    replicas are handed out with stream=None and callers just run forward
    passes normally - this also means pool_size on CPU only helps to the
    extent the CPU has spare cores, unrelated to the CUDA-stream story.
    """

    def __init__(self, base_model: "ChessTransformer", pool_size: int, device: str):
        self.device = device
        self._pool: "queue.Queue" = queue.Queue()
        is_cuda = device.startswith("cuda")
        for _ in range(pool_size):
            replica = copy.deepcopy(base_model).to(device)
            replica.eval()  # disable dropout during self-play data generation
            stream = torch.cuda.Stream(device=device) if is_cuda else None
            self._pool.put((replica, stream))

    @contextlib.contextmanager
    def acquire(self):
        """Block until a replica is free, yield (replica, stream), return
        it to the pool when the caller is done."""
        replica, stream = self._pool.get()
        try:
            yield replica, stream
        finally:
            self._pool.put((replica, stream))


@contextlib.contextmanager
def _acquire_pair(white_pool: ModelPool, black_pool: ModelPool):
    """
    Acquire one replica for white and one for black, for the duration of
    a whole game. If white_pool and black_pool are the SAME pool (i.e.
    self-play with one set of weights playing both sides), only a single
    replica is checked out and used for both colors - no need to hold two
    instances of identical weights for one game.
    """
    if white_pool is black_pool:
        with white_pool.acquire() as (replica, stream):
            yield replica, stream, replica, stream
    else:
        with white_pool.acquire() as (w_replica, w_stream):
            with black_pool.acquire() as (b_replica, b_stream):
                yield w_replica, w_stream, b_replica, b_stream


def _run_on_replica(replica: "ChessTransformer", stream, input_tensor: torch.Tensor, board: chess.Board, temperature: float):
    """
    Run one move's worth of GPU work (the forward pass AND select_move_
    from_output's per-legal-move indexing/softmax/sampling) on the given
    replica's dedicated stream, then synchronize that stream before
    returning - so the CPU-side result is safe to use immediately without
    racing the GPU. If stream is None (CPU), just runs normally.
    """
    if stream is not None:
        with torch.cuda.stream(stream):
            with torch.no_grad():
                policy_logits, _value = replica(input_tensor)
            move = select_move_from_output(policy_logits[0], board, temperature=temperature)
        stream.synchronize()
        return move
    else:
        with torch.no_grad():
            policy_logits, _value = replica(input_tensor)
        return select_move_from_output(policy_logits[0], board, temperature=temperature)



def play_game(
    white_pool: ModelPool,
    black_pool: ModelPool,
    device: str = "cpu",
    temperature: float = 1.0,
    **kwargs
):
    """
    Play one full self-play game, using replicas checked out from
    white_pool / black_pool for the entire game's duration (including
    across move-cap retries - see ModelPool docstring for why replicas
    exist and _acquire_pair for the same-pool self-play case).

    Returns:
        history: list of (input_tensor, mover_color) for every ply played
        outcome: dict mapping chess.WHITE / chess.BLACK -> result in {-1.0, 0.0, 1.0}
                    - 0.0 for both sides if the game results in a draw
    """
    with _acquire_pair(white_pool, black_pool) as (white_replica, white_stream, black_replica, black_stream):
        attempt = 0
        while True:
            attempt += 1
            board = chess.Board()
            history = []

            full_move_count = 0
            while not board.is_game_over(claim_draw=True) and full_move_count < MAX_FULL_MOVES:
                input_tensor = board_to_tensor(board).unsqueeze(0).to(device)

                if board.turn == chess.WHITE:
                    replica, stream = white_replica, white_stream
                else:
                    replica, stream = black_replica, black_stream

                move = _run_on_replica(replica, stream, input_tensor, board, temperature)

                history.append({
                    "tensor": input_tensor.squeeze(0).cpu(),
                    "mover": board.turn,
                    "board_fen": board.fen(),  # position BEFORE the move, for
                                                # reconstructing legal moves /
                                                # plane indices during training
                    "move_uci": move.uci(),
                })
                board.push(move)

                if board.turn == chess.WHITE:
                    # a full move just completed (black just moved)
                    full_move_count += 1

            game = chess.pgn.Game.from_board(board)

            game.headers["Event"] = f"Epoch {kwargs['Epoch']} Game {kwargs['Count']} Attempt {attempt}"
            game.headers["White"] = kwargs["White"]
            game.headers["Black"] = kwargs["Black"]
            game.headers["Result"] = board.result()

            os.makedirs(f"{kwargs['Epoch']}", exist_ok=True)
            with open(f"{kwargs['Epoch']}/epoch{kwargs['Epoch']}game{kwargs['Count']}attempt{attempt}.pgn", "w", encoding="utf-8") as pgn_file:
                pgn_file.write(str(game))

            if board.is_game_over(claim_draw=True):
                if board.is_fifty_moves() or board.is_fivefold_repetition() or board.is_seventyfive_moves() or board.is_fifty_moves() or board.is_repetition():
                    result = "0-0"
                else:
                    result = board.result(claim_draw=True)
            else:
                # hit the 200-full-move cap without a natural conclusion
                continue

            break

    if result == "1-0":
        outcome = {chess.WHITE: 1.0, chess.BLACK: -1.0}
    elif result == "0-1":
        outcome = {chess.WHITE: -1.0, chess.BLACK: 1.0}
    elif result == "0-0":
        outcome = {chess.WHITE: -1.0, chess.BLACK: -1.0}
    else:
        outcome = {
            chess.WHITE: 0.0 if board.turn == chess.BLACK else -0.1, # White last moved, causing the draw
            chess.BLACK: 0.0 if board.turn == chess.WHITE else -0.1  # Black last moved, causing the draw
        }

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
    threads: int = 1,
    model_pool_size: int = None,
):
    if os.path.isdir(f"{model_white.epoch}"):
        print_info(f"Removing old games path at `{model_white.epoch}/`")
        # topdown=False is critical: it clears files/subfolders before the parent folder
        for root, dirs, files in os.walk(f"{model_white.epoch}", topdown=False):
            # 1. Delete all individual files
            for f in files:
                os.remove(os.path.join(root, f))
            
            # 2. Delete all now-empty subdirectories
            for directory in dirs:
                os.rmdir(os.path.join(root, directory))

        # 3. Delete the top-level directory itself
        os.rmdir(f"{model_white.epoch}")
    """
    Play games_per_epoch self-play games (up to `threads` games running
    concurrently), then train each model on the resulting occurrences
    (see module docstring for the value + policy loss this applies).

    Parallelism: for the self-play phase only, `model_pool_size` replicas
    of model_white (and, if model_black is a different model, model_black
    too) are created, each bound to its own CUDA stream - see ModelPool's
    docstring for why. model_pool_size defaults to `threads`, so by
    default every concurrently-running game gets its own replica; passing
    a smaller model_pool_size makes extra threads queue for a free replica
    (still correct, just more contention). The replica pools are
    discarded once self-play finishes - training afterward uses
    model_white/model_black directly, sequentially, exactly as before.
    threads=1 (with the default pool size) preserves the original
    fully-sequential behavior.
    """
    if model_pool_size is None:
        model_pool_size = threads

    same_weights = model_white is model_black
    white_pool = ModelPool(model_white, model_pool_size, device)
    black_pool = white_pool if same_weights else ModelPool(model_black, model_pool_size, device)

    # per-occurrence, NOT deduplicated - see module docstring for why
    all_occurrences = {chess.WHITE: [], chess.BLACK: []}

    summary = {'white': 0, 'black': 0, 'draw': 0, 'attempts': 0, 'states': 0}

    def run_one_game(game_idx):
        return play_game(
            white_pool, black_pool, device=device, temperature=temperature,
            Epoch=epoch, White=f"Generation {model_white.epoch}",
            Black=f"Generation {model_black.epoch}", Count=game_idx + 1,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(run_one_game, game_idx) for game_idx in range(games_per_epoch)]

        for future in concurrent.futures.as_completed(futures):
            history, outcome, attempts = future.result()

            for entry in history:
                mover = entry["mover"]
                all_occurrences[mover].append({
                    "tensor": entry["tensor"],
                    "board_fen": entry["board_fen"],
                    "move_uci": entry["move_uci"],
                    "outcome": outcome[mover],
                })

            if outcome[chess.WHITE] == 1: summary['white'] += 1
            elif outcome[chess.WHITE] == -1: summary['black'] += 1
            else: summary['draw'] += 1
            summary['attempts'] += attempts

    summary['states'] = len(all_occurrences[chess.WHITE]) + len(all_occurrences[chess.BLACK])

    # self-play is done - drop the replica pools before training, so the
    # (potentially large) training batches aren't competing with them for
    # VRAM. The caching allocator would likely reuse this memory anyway,
    # but empty_cache() makes it explicit rather than relying on that.
    del white_pool, black_pool
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    average_loss_white_model = train_on_occurrences(
        model_white, all_occurrences[chess.WHITE], optimizer, device=device, train_batch_size=train_batch_size
    )
    average_loss_black_model = train_on_occurrences(
        model_black, all_occurrences[chess.BLACK], optimizer, device=device, train_batch_size=train_batch_size
    )

    return (average_loss_white_model, average_loss_black_model, summary)


def train_on_occurrences(model, occurrences, optimizer: torch.optim.Optimizer, device: str = "cpu", train_batch_size: int = 256):
    """
    Train `model` on a list of occurrence dicts (tensor, board_fen,
    move_uci, outcome), combining:
        - value loss: MSE(value_pred, outcome)
        - policy loss: advantage-weighted NLL of the move actually chosen,
          using the value head's own (detached) prediction as the baseline

    Both losses are backpropagated together per batch. Note the per-move
    policy loss requires reconstructing each occurrence's legal-move list
    from its stored FEN and re-running move_to_plane_index over it, so
    this is noticeably more expensive per batch than plain value-only
    training was - if that becomes a bottleneck on CPU, reducing
    train_batch_size or games_per_epoch is the easiest lever.
    """
    if not occurrences:
        return 0.0

    model.train()
    total_loss = 0.0
    num_batches = 0

    indices = list(range(len(occurrences)))
    random.shuffle(indices)

    for start in range(0, len(indices), train_batch_size):
        batch_idx = indices[start:start + train_batch_size]
        batch = [occurrences[i] for i in batch_idx]

        batch_x = torch.stack([o["tensor"] for o in batch]).to(device)
        batch_y = torch.tensor([o["outcome"] for o in batch], dtype=torch.float32, device=device)

        optimizer.zero_grad()
        policy_logits, value_preds = model(batch_x)  # (B, 8, 8, 73), (B,)

        value_loss = F.mse_loss(value_preds, batch_y)

        # advantage: how much better/worse the actual outcome was than the
        # value head already expected. Detached so this only acts as a
        # fixed per-sample weight for the policy loss, not a second
        # gradient path back through the value head.
        advantages = batch_y - value_preds.detach()

        policy_losses = []
        for i, occurrence in enumerate(batch):
            board = chess.Board(occurrence["board_fen"])
            mover = board.turn
            legal_moves = list(board.legal_moves)

            move_logits = []
            chosen_idx = None
            for j, mv in enumerate(legal_moves):
                from_row, from_col = _canonical_rc(mv.from_square, mover)
                plane_idx = move_to_plane_index(mv, board)
                move_logits.append(policy_logits[i, from_row, from_col, plane_idx])
                if mv.uci() == occurrence["move_uci"]:
                    chosen_idx = j

            move_logits = torch.stack(move_logits)
            log_probs = F.log_softmax(move_logits, dim=0)
            policy_losses.append(-log_probs[chosen_idx] * advantages[i])

        policy_loss = torch.stack(policy_losses).mean()

        loss = value_loss + policy_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)

from datetime import datetime
def print_info(*args):
    time_str = str(datetime.now().strftime("%H:%M:%S"))
    print(f"{time_str:<12}", *args)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_info(f"Using device `{device}`")
    model = ChessTransformer(0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    start_epoch = None
    guess_epoch = 0
    while os.path.exists(f"model{guess_epoch}.pth"):
        guess_epoch += 1
    if len(sys.argv) >= 2:
        start_epoch = int(sys.argv[1])
        print_info(f"Attempting to load at Epoch {start_epoch}")
        model.load_state_dict(torch.load(f"model{start_epoch}.pth", weights_only=True))
        model.epoch = start_epoch
        print_info(f"Successfully loaded Epoch {start_epoch}")
    else:
        print_info(f"No model number provided. Starting at Epoch {guess_epoch}")
        start_epoch = guess_epoch - 1

    def print_summary(summary):
        print_info(f"\tEpoch Game Summary")
        print_info(f"\tAttempts:     {summary['attempts']}")
        print_info(f"\tBoard States: {summary['states']}")
        print_info(f"\tWhite Wins:   {summary['white']}")
        print_info(f"\tBlack Wins:   {summary['black']}")
        print_info(f"\tDraws:        {summary['draw']}")

    for epoch in range(0 if not start_epoch else start_epoch + 1, 200):
        model.epoch = epoch

        print()
        print_info(f"Beginning Epoch {epoch}")
        
        games_this_epoch = min(1024, 64 * (2**epoch))
        avg_loss_white, avg_loss_black, summary = play_epoch(
            model, model, optimizer, epoch, device=device, games_per_epoch=games_this_epoch, train_batch_size=games_this_epoch // 4, threads=8
        )
        
        print_info(f"Epoch {epoch} complete. Average value loss white: {avg_loss_white:.4f} Average value loss black: {avg_loss_black:.4f}")
        print_summary(summary)
        print_info(f"Saving Epoch Model as `model{epoch}.pth`")
        torch.save(model.state_dict(), f"model{epoch}.pth")