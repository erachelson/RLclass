import numpy as np

class tictactoe:
    def __init__(self):
        self.current_player = 1 # 1 or 2
        # self.actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.board = np.zeros((9,),dtype=int) # [0,0,0,0,0,0,0,0,0]
        self.empty_cells = 9
        # self.nb_move = 0
        self.game_over = False # -1: running, O: nul, 1: player 1 win, 2: player 2 win
    
    def reset(self, board=np.zeros((9,),dtype=int)):
        self.board = board
        self.empty_cells = 0
        for i in range(9):
            if self.board[i]==0:
                self.empty_cells+=1
        self.current_player = 1
        if self.empty_cells %2 != 0:
            self.current_player = 2
        self.game_over = self.empty_cells==0
    
    def change_player(self):
        if self.current_player==1:
            self.current_player=2
        else:
            self.current_player=1
    
    def legal_moves(self):
        valid = []
        for a in range(9):
            if self.board[a]==0:
                valid.append(a)
        return np.array(valid)
    
    def next_board(self, a):
        new_board = self.board
        if new_board[a]==0:
            new_board[a] = self.current_player
        else:
            print("Invalid action ", a, ", it's still player", self.current_player, "'s turn")
        return new_board
    
    def play(self, a):
        if self.board[a]==0:
            self.board[a] = self.current_player
            self.empty_cells = self.empty_cells-1
            if self.empty_cells==0:
                self.game_over=True
            self.change_player()
        else:
            print("Invalid action ", a, ", it's still player", self.current_player, "'s turn")
    
    def winner(self):
        winner=0
        for player in range(1,3):
            if (   self.board[0] == self.board[1] == self.board[2] == player 
                or self.board[3] == self.board[4] == self.board[5] == player 
                or self.board[6] == self.board[7] == self.board[8] == player 
                or self.board[0] == self.board[3] == self.board[6] == player 
                or self.board[1] == self.board[4] == self.board[7] == player 
                or self.board[2] == self.board[5] == self.board[8] == player 
                or self.board[0] == self.board[4] == self.board[8] == player 
                or self.board[6] == self.board[4] == self.board[2] == player ):
                winner = player
        return winner
    
    def board_str(self):
        s = ""
        for i in range(3):
            for j in range(3):
                a = 3*i+j
                if self.board[a]==0:
                    s+='.'
                elif self.board[a]==1:
                    s+='X'
                elif self.board[a]==2:
                    s+='O'
            if(i<2):
                s +="\n"
        return s
