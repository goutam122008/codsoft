import sys

# Quick win condition map (indices on a 0-8 board)
WIN_STATES = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),  # Rows
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),  # Cols
    (0, 4, 8),
    (2, 4, 6),  # Diagonals
]


def check_winner(b):
    for p1, p2, p3 in WIN_STATES:
        if b[p1] == b[p2] == b[p3] and b[p1] != " ":
            return b[p1]
    return None


def minimax(b, is_max):
    winner = check_winner(b)
    if winner == "O":
        return 1
    if winner == "X":
        return -1
    if " " not in b:
        return 0

    scores = []
    for i in range(9):
        if b[i] == " ":
            b[i] = "O" if is_max else "X"
            scores.append(minimax(b, not is_max))
            b[i] = " "  # undo

    return max(scores) if is_max else min(scores)


def get_ai_move(b):
    best_score = -2
    best_move = None
    for i in range(9):
        if b[i] == " ":
            b[i] = "O"
            score = minimax(b, False)
            b[i] = " "
            if score > best_score:
                best_score = score
                best_move = i
    return best_move


def draw_board(b):
    print(f"\n {b[0]} | {b[1]} | {b[2]} \n---|---|---\n {b[3]} | {b[4]} | {b[5]} \n---|---|---\n {b[6]} | {b[7]} | {b[8]} \n")


def main():
    board = [" "] * 9
    print("--- Unbeatable Tic-Tac-Toe (You: X, AI: O) ---")
    print("Positions: 1-9")

    while " " in board and not check_winner(board):
        draw_board(board)
        try:
            move = int(input("Your move (1-9): ")) - 1
            if move < 0 or move > 8 or board[move] != " ":
                print("Invalid move, spot taken or out of bounds.")
                continue
            board[move] = "X"
        except (ValueError, IndexError):
            print("Enter a number between 1 and 9.")
            continue
        except (KeyboardInterrupt, EOFError):
            sys.exit("\nGame quit.")

        if " " not in board or check_winner(board):
            break

        print("AI calculating...")
        board[get_ai_move(board)] = "O"

    draw_board(board)
    winner = check_winner(board)
    if winner == "O":
        print("AI wins. Gg!")
    elif winner == "X":
        print("You won?! (Shouldn't happen!)")
    else:
        print("It's a draw!")


if __name__ == "__main__":
    main()