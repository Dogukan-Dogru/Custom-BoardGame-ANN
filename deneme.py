import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

GRID_SIZE = 7


def print_board(player1_pieces, player2_pieces):
    print("   ", end="")
    for i in range(1, GRID_SIZE + 1):
        print(f"{i:2}", end=" ")
    print()

    for i in range(GRID_SIZE):
        print(f"{chr(97 + i):2} ", end="")
        for j in range(1, GRID_SIZE + 1):
            if (i, j - 1) in player1_pieces:
                print('X', end=' ')
            elif (i, j - 1) in player2_pieces:
                print('O', end=' ')
            else:
                print('--', end=' ')
        print()


def initialize_pieces(num_pieces):
    pieces = set()
    while len(pieces) < num_pieces:
        x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
        pieces.add((x, y))
    return pieces


def get_valid_moves(piece, player_pieces, opponent_pieces):
    x, y = piece
    moves = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_moves = [(nx, ny) for nx, ny in moves if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE]
    valid_moves = [(nx, ny) for nx, ny in valid_moves if (nx, ny) not in opponent_pieces]
    valid_moves = [(nx, ny) for nx, ny in valid_moves if (nx, ny) not in player_pieces]
    return valid_moves


def user_move(player_pieces, opponent_pieces):
    while True:
        print("Your turn")
        piece_input = input("Choose piece to move (e.g., c4): ")
        if len(piece_input) != 2 or not piece_input[0].isalpha() or not piece_input[1].isdigit():
            print("Invalid input. Please enter a valid piece coordinate.")
            continue

        x, y = ord(piece_input[0]) - ord('a'), int(piece_input[1]) - 1
        current_piece = (x, y)

        if current_piece not in player_pieces:
            print("Invalid piece coordinate. Please choose a piece that belongs to you.")
            continue

        valid_moves = get_valid_moves(current_piece, player_pieces, opponent_pieces)
        print("Valid moves:", [f"{chr(97 + nx)}{ny + 1}" for nx, ny in valid_moves])

        if not valid_moves:
            print("No valid moves for the selected piece. Choose another piece.")
            continue

        new_position_input = input(f"Choose the new position for {piece_input}: ")
        if len(new_position_input) != 2 or not new_position_input[0].isalpha() or not new_position_input[1].isdigit():
            print("Invalid input. Please enter a valid new position coordinate.")
            continue

        nx, ny = ord(new_position_input[0]) - ord('a'), int(new_position_input[1]) - 1
        new_position = (nx, ny)

        if new_position not in valid_moves:
            print("Invalid move. The chosen new position is not valid for the selected piece.")
            continue

        player_pieces.remove(current_piece)
        player_pieces.add(new_position)
        print(f"Player moves the piece at {piece_input} to {new_position_input}")
        return new_position


def ai_move(neural_network, ai_pieces, player_pieces):
    print("Computer's turn")

    # Prepare the input data for the neural network
    board_state = np.zeros((GRID_SIZE, GRID_SIZE, 1), dtype=np.float32)
    for piece in ai_pieces:
        board_state[piece[0], piece[1], 0] = 1  # Set a value to indicate AI's pieces

    # Reshape the input to match the expected input shape of the neural network
    input_data = np.expand_dims(board_state, axis=0)

    # Predict the move probabilities using the neural network
    move_probabilities = neural_network.predict(input_data)[0]

    # Get valid moves
    valid_moves = [idx for idx, _ in enumerate(move_probabilities) if
                   get_move(idx) in get_valid_moves(piece, ai_pieces, player_pieces)]

    if not valid_moves:
        print("No valid moves. User wins!")
        return None

    # Choose a move based on the probabilities
    chosen_move_idx = \
    np.random.choice(valid_moves, 1, p=move_probabilities[valid_moves] / sum(move_probabilities[valid_moves]))[0]
    chosen_move = get_move(chosen_move_idx)

    # Convert the chosen move to coordinates
    current_piece = (ord(chosen_move[0]) - ord('a'), int(chosen_move[1]) - 1)

    # Get a valid new position for the chosen move
    new_position = random.choice(get_valid_moves(current_piece, ai_pieces, player_pieces))

    ai_pieces.remove(current_piece)
    ai_pieces.add(new_position)

    print(f"Computer moves the piece at {chosen_move} to {get_move(new_position)}")
    return new_position


def get_move(index):
    # Convert an index to a move representation, e.g., 0 -> 'a1', 1 -> 'a2', ..., 48 -> 'g7'
    row = index // GRID_SIZE
    col = index % GRID_SIZE
    return f"{chr(97 + col)}{row + 1}"


def main():
    # Number of pieces for each player and maximum turns
    num_pieces = int(input("Enter the number of pieces for each player: "))
    turn_limit = int(input("Enter the maximum number of turns: "))

    # Choose the player
    user_choice = input("Do you want to be Player 1 (X) or Player 2 (O)? ").lower()

    # Create the neural network for the AI
    input_shape = (GRID_SIZE, GRID_SIZE, 1)
    output_size = GRID_SIZE * GRID_SIZE  # Number of possible moves
    neural_network = create_neural_network(input_shape, output_size)

    if user_choice == '1':
        player1_pieces = initialize_pieces(num_pieces)
        player2_pieces = initialize_pieces(num_pieces)
        user_turn = True
    else:
        player1_pieces = initialize_pieces(num_pieces)
        player2_pieces = initialize_pieces(num_pieces)
        user_turn = False

    print("Initial Board:")
    print_board(player1_pieces, player2_pieces)

    # Game loop
    for turn in range(1, turn_limit + 1):
        print(f"\nTurn {turn}/{turn_limit}")

        if user_turn:
            move_result = user_move(player1_pieces, player2_pieces)
            if move_result is None:
                break
        else:
            move_result = ai_move(neural_network, player2_pieces, player1_pieces)
            if move_result is None:
                break

        print_board(player1_pieces, player2_pieces)
        user_turn = not user_turn

    print("\nGame Over!")


if __name__ == "__main__":
    main()
