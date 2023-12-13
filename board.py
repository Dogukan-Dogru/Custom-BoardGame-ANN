import tkinter as tk
from tkinter import ttk
import random

class GameBoardUI:
    def __init__(self, master, rows, cols):
        self.master = master
        self.rows = rows
        self.cols = cols
        self.create_board()

    def create_board(self):
        # Create labels for columns on the top
        for col in range(self.cols):
            label = f"{col + 1}"
            ttk.Label(self.master, text=label, width=3, padding=(5, 2)).grid(row=0, column=col + 1, sticky='nsew')

        # Create labels for rows on the left
        for row in range(self.rows):
            label = f"{chr(ord('A') + row)}"
            ttk.Label(self.master, text=label, width=3, padding=(5, 2)).grid(row=row + 1, column=0, sticky='nsew')

        buttons = [ttk.Button(self.master, style='GameButton.TButton',
                               command=lambda row=row, col=col: self.button_click(row, col))
                   for row in range(self.rows) for col in range(self.cols)]

        # Place buttons on the grid
        for idx, button in enumerate(buttons):
            button.grid(row=(idx // self.cols) + 1, column=(idx % self.cols) + 1, sticky='nsew')
            self.master.grid_rowconfigure((idx // self.cols) + 1, weight=1)
            self.master.grid_columnconfigure((idx % self.cols) + 1, weight=1)

        # Shuffle buttons randomly
        random.shuffle(buttons)

        # Assign pieces to shuffled buttons
        pieces = ["X"] * 5 + ["O"] * 5
        for button, piece in zip(buttons, pieces):
            button.config(text=piece)

    def button_click(self, row, col):
        label = f"{chr(ord('A') + row)}{col + 1}"
        print(f"Button clicked at {label}")

def main():
    rows = 7
    cols = 7

    root = tk.Tk()
    root.title("Game Board")

    style = ttk.Style()
    style.configure('GameButton.TButton', font=('Helvetica', 10), padding=(10, 5), width=4, height=2)

    game_board_ui = GameBoardUI(root, rows, cols)

    root.mainloop()

if __name__ == "__main__":
    main()
