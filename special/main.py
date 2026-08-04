import chess
import chess.pgn
import os

file_path = input('Path to .pgn: ')

pgn = open(file_path)
game = chess.pgn.read_game(pgn)
board = game.board()

for move in game.mainline_moves():
    board.push(move)
    print(board)
    input()
    os.system('clear')
