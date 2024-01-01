import random
import numpy as np

GRID_SIZE = 7
xcount = 0
ocount = 0

def print_board(player1_pieces, player2_pieces):
    global xcount
    global ocount
    print("   ", end="")
    for i in range(1, GRID_SIZE + 1):
        print(f" {i:2}", end="   ")
    print()

    for i in range(GRID_SIZE):
        print(f"{chr(97 + i):2} ", end="")
        for j in range(1, GRID_SIZE + 1):
            if (i, j - 1) in player1_pieces:
                print('  X', end='   ')
                xcount += 1
            elif (i, j - 1) in player2_pieces:
                print('  O', end='   ')
                ocount += 1
            else:
                print(' ---', end='  ')
        print()
    print("ocount=",ocount)

    print("xcount=", xcount)

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state, explore=True):
        if explore and random.uniform(0, 1) < self.exploration_prob:
            return random.choice(range(self.action_size))
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state, :])
        self.q_table[state, action] = (1 - self.learning_rate) * self.q_table[state, action] + \
                                      self.learning_rate * (reward + self.discount_factor * self.q_table[next_state, best_next_action])

def initialize_pieces(num_pieces):
    pieces = set()

    while len(pieces) < num_pieces:
        x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
        current_piece = (x, y)

        # Check if the coordinate is already in the set
        if current_piece in pieces:
            continue  # Skip this iteration if the coordinate is not unique

        pieces.add(current_piece)

    return pieces

def get_valid_moves(piece, player_pieces, opponent_pieces):
    if piece is None:
        return []  # No valid moves if the piece is None

    x, y = piece
    moves = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_moves = [(nx, ny) for nx, ny in moves if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE]

    # Filter out moves to squares occupied by the opponent
    valid_moves = [(nx, ny) for nx, ny in valid_moves if (nx, ny) not in opponent_pieces]

    # Filter out moves to squares occupied by the player's own pieces
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

def ai_move_train(ai_pieces, player_pieces):
    print("Computer's turn")
    current_piece = random.choice(list(ai_pieces))

    valid_moves = get_valid_moves(current_piece, ai_pieces, player_pieces)
    print("Valid moves:", [f"{chr(97 + nx)}{chr(49 + ny)}" for nx, ny in valid_moves])

    if not valid_moves:
        print("No valid moves. User wins!")
        return None

    new_position = random.choice(valid_moves)
    ai_pieces.remove(current_piece)
    ai_pieces.add(new_position)

    # Print the move
    current_position_str = f"{chr(98 + current_piece[1])}{chr(49 + current_piece[0])}"
    new_position_str = f"{chr(98 + new_position[1])}{chr(49 + new_position[0])}"
    print(f"Computer moves the piece at {current_position_str} to {new_position_str}")

    return new_position
def ai_move(ai_pieces, player_pieces, agent):
    print("Computer's turn")

    # Convert the set of AI's pieces to a list for iteration
    ai_pieces_list = list(ai_pieces)

    # Initialize variables to keep track of the best move and its expected reward
    best_move = None
    best_reward = float('-inf')  # Negative infinity, to ensure any reward will be an improvement

    # Iterate over AI's pieces to find the best move
    for current_piece in ai_pieces_list:
        # Get the state representation for the current piece
        state = len(ai_pieces)

        # Iterate over valid moves for the current piece
        valid_moves = get_valid_moves(current_piece, ai_pieces, player_pieces)
        for new_position in valid_moves:
            # Get the action index corresponding to the move
            action = valid_moves.index(new_position)

            # Use the Q-value for the current state and action
            current_reward = agent.q_table[state, action]

            # Count the number of valid moves for the player after the AI's move
            future_player_moves = sum(len(get_valid_moves(piece, player_pieces - {current_piece}, ai_pieces | {new_position})) for piece in player_pieces)

            # Update the best move if the current reward is better and minimizes the future player moves
            if current_reward > best_reward or (current_reward == best_reward and future_player_moves < best_player_moves):
                best_reward = current_reward
                best_move = (current_piece, new_position)
                best_player_moves = future_player_moves

    # Check if there are valid moves
    if best_move is None:
        print("No valid moves. User wins!")
        return None

    # Update AI's pieces based on the best move
    current_piece, new_position = best_move
    ai_pieces.remove(current_piece)
    ai_pieces.add(new_position)

    # Print the move
    current_position_str = f"{chr(97 + current_piece[1])}{chr(49 + current_piece[0])}"
    new_position_str = f"{chr(97 + new_position[1])}{chr(49 + new_position[0])}"
    print(f"Computer moves the piece at {current_position_str} to {new_position_str}")

    return new_position



def determine_winner_by_space(player1_pieces, player2_pieces):
    player1_valid_moves = sum(len(get_valid_moves(piece, player1_pieces, player2_pieces)) for piece in player1_pieces)
    player2_valid_moves = sum(len(get_valid_moves(piece, player2_pieces, player1_pieces)) for piece in player2_pieces)

    if player1_valid_moves > player2_valid_moves:
        return "Player 1 (User) wins based on valid movable space!"
    elif player2_valid_moves > player1_valid_moves:
        return "Player 2 (Computer) wins based on valid movable space!"
    else:
        return "It's a draw based on valid movable space!"

#def validate_board(user_choice,num_pieces):


def main():
    num_pieces = int(input("Enter the number of pieces for each player: "))
    turn_limit = int(input("Enter the maximum number of turns: "))
    num_episodes = 3000

    user_choice = input("Do you want to be Player 1 (X) or Player 2 (O)? ").lower()

    for episode in range(1, num_episodes + 1):
        print(f"\nEpisode {episode}/{num_episodes}")
        print("Validating...")
        global xcount
        global ocount

        while True:
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

            if ocount != xcount:
                print("Recreating the board...")
                if user_choice == '1':
                    player1_pieces = initialize_pieces(num_pieces)
                    player2_pieces = initialize_pieces(num_pieces)
                else:
                    player1_pieces = initialize_pieces(num_pieces)
                    player2_pieces = initialize_pieces(num_pieces)

                print("Initial Board:")
                print_board(player1_pieces, player2_pieces)
                xcount = 0
                ocount = 0
                continue
            break
        # Create Q-learning agent
        agent = QLearningAgent(state_size=num_pieces * 2, action_size=4)

        for turn in range(1, turn_limit + 1):
            print(f"\nTurn {turn}/{turn_limit}")

            if user_turn:
                if episode == num_episodes:  # Play with the user only in the last episode
                    move_result = user_move(player1_pieces, player2_pieces)
                else:
                    # AI's turn during training
                    new_position = ai_move(player1_pieces, player2_pieces,agent)
                    state = len(get_valid_moves(new_position, player1_pieces,player2_pieces))
                    action = agent.choose_action(state,True)
                    reward = len(get_valid_moves(new_position, player1_pieces,player2_pieces))  # Reward is the number of possible moves

                    next_state = len(get_valid_moves(new_position, player1_pieces,player2_pieces))
                    agent.update_q_table(state, action, reward, next_state)

                    print("State:", state)
                    print("Action:", action)
                    print("New Position:", new_position)
                    print("Reward:", reward)
                    print("Next State:", next_state)

                    if new_position is None:
                        break
            else:
                # AI's turn during user play
                new_position = ai_move(player2_pieces, player1_pieces,agent)
                state = len(get_valid_moves(new_position, player2_pieces,player1_pieces))
                action = agent.choose_action(state,False)
                reward = len(get_valid_moves(new_position, player2_pieces,player1_pieces))  # Reward is the number of possible moves

                next_state = len(get_valid_moves(new_position, player2_pieces,player1_pieces))
                agent.update_q_table(state, action, reward, next_state)

                print("State:", state)
                print("Action:", action)
                print("New Position:", new_position)
                print("Reward:", reward)
                print("Next State:", next_state)

                if new_position is None:
                    break

            print_board(player1_pieces, player2_pieces)
            user_turn = not user_turn

        if episode < num_episodes:
            print("Training AI...")

    winner_result = determine_winner_by_space(player1_pieces, player2_pieces)
    print(f"\nFinal Board: {winner_result}")

if __name__ == "__main__":
    main()
