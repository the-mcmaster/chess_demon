"""
engine.py

UCI protocol frontend for ChessTransformer. Position is set by the GUI;
each `go` runs one model forward pass and returns bestmove (no search tree).
"""

import os
import sys

import chess
import torch

from model import ChessTransformer, board_to_tensor
from training import select_move_from_output

ENGINE_NAME = "chess_demon"
ENGINE_AUTHOR = "Eric Ovenden"


def _uci_print(line: str) -> None:
    print(line, flush=True)


def resolve_checkpoint(arg: str | None) -> tuple[str, int]:
    """
    Return (path, epoch) for the weights to load.

    arg may be an epoch int, a path to a .pth file, or None (highest modelN.pth).
    """
    if arg is not None:
        if arg.endswith(".pth") or os.path.isfile(arg):
            path = arg
            basename = os.path.basename(path)
            # model12.pth -> epoch 12 when the name fits the training convention
            epoch = 0
            if basename.startswith("model") and basename.endswith(".pth"):
                mid = basename[len("model") : -len(".pth")]
                if mid.isdigit():
                    epoch = int(mid)
            return path, epoch
        epoch = int(arg)
        path = f"model{epoch}.pth"
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path, epoch

    guess_epoch = 0
    while os.path.exists(f"model{guess_epoch}.pth"):
        guess_epoch += 1
    if guess_epoch == 0:
        raise FileNotFoundError("No model{N}.pth checkpoints found in working directory")
    epoch = guess_epoch - 1
    return f"model{epoch}.pth", epoch


def load_model(checkpoint_path: str, epoch: int, device: str) -> ChessTransformer:
    model = ChessTransformer(epoch).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def choose_move(
    board: chess.Board,
    model: ChessTransformer,
    device: str,
) -> tuple[chess.Move | None, float]:
    """
    Run the network on the side-to-move position.
    Returns (best_move_or_None, value in [-1, 1] for the mover).
    """
    if board.is_game_over(claim_draw=True) or not any(board.legal_moves):
        return None, 0.0

    x = board_to_tensor(board).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, value = model(x)
    move = select_move_from_output(policy_logits[0], board, temperature=0.0)
    return move, float(value[0].item())


def parse_position(tokens: list[str], board: chess.Board) -> None:
    """Apply a UCI `position` command's tokens (without the leading `position`)."""
    if not tokens:
        return

    if tokens[0] == "startpos":
        board.reset()
        i = 1
        if i < len(tokens) and tokens[i] == "moves":
            i += 1
            for mv in tokens[i:]:
                board.push_uci(mv)
        return

    if tokens[0] == "fen":
        # FEN is six fields; stop at "moves" or end of line
        fen_parts: list[str] = []
        i = 1
        while i < len(tokens) and tokens[i] != "moves":
            fen_parts.append(tokens[i])
            i += 1
            if len(fen_parts) >= 6:
                break
        board.set_fen(" ".join(fen_parts))
        if i < len(tokens) and tokens[i] == "moves":
            i += 1
            for mv in tokens[i:]:
                board.push_uci(mv)
        return


def handle_go(board: chess.Board, model: ChessTransformer, device: str) -> None:
    move, value = choose_move(board, model, device)
    # value is mover-relative in [-1, 1]; surface as centipawns for GUIs
    cp = int(round(value * 1000))
    _uci_print(f"info score cp {cp} depth 1")
    if move is None:
        _uci_print("bestmove 0000")
    else:
        _uci_print(f"bestmove {move.uci()}")


def uci_loop(model: ChessTransformer, device: str) -> None:
    board = chess.Board()

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue

        tokens = line.split()
        cmd = tokens[0]

        if cmd == "uci":
            _uci_print(f"id name {ENGINE_NAME}")
            _uci_print(f"id author {ENGINE_AUTHOR}")
            _uci_print("uciok")

        elif cmd == "isready":
            _uci_print("readyok")

        elif cmd == "ucinewgame":
            board.reset()

        elif cmd == "position":
            parse_position(tokens[1:], board)

        elif cmd == "go":
            # wtime/btime/depth/movetime accepted but ignored (no search tree)
            handle_go(board, model, device)

        elif cmd == "stop":
            pass

        elif cmd == "ponderhit":
            pass

        elif cmd == "setoption":
            pass

        elif cmd == "quit":
            return

        # Unknown commands are ignored (UCI-tolerant)


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) >= 2 else None
    path, epoch = resolve_checkpoint(arg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(path, epoch, device)
    uci_loop(model, device)


if __name__ == "__main__":
    main()
