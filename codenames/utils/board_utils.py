from codenames.board import Board

def make_board_set(num_boards, seed):
    boards = []
    for i in range(num_boards):
        board = Board(seed=seed + i)
        boards.append([board.goal_words, board.avoid_words, board.neutral_words])
    return boards