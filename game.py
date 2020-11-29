import numpy as np
import math

class Game:
    def __init__(self, r = 6, c = 6, num_of_mines = 6, random_assign = True):        
        """
        specify the num_of_mines by Game(r = num_of _row, c = num_of_col, num_of_mines = num_of_mines, random_assign = False)
        randomly assign num_of_mines by Game(r = num_of _row, c = num_of_col, random_assign = True)
        """
        self.r = r
        self.c = c
        self.random_assign = random_assign
        if random_assign:
            self.num_of_mines = np.random.randint(math.sqrt(r * c), math.floor(r * c / 2) + 1)
        else:
            self.num_of_mines = num_of_mines
        
        ## game state
        self.state_mask = np.zeros((self.r, self.c), dtype = int)

        ## num of grids has been uncovered
        ## when num_uncover == num_of_mines, the player wins the game
        self.num_uncover = r * c

        ## an int rxc 2d np array recording where is the mine(1)
        self.mines = np.zeros((self.r, self.c), dtype = int) 

        ## an int rxc 2d np array recording what the number to display when a grid is revealed if the grid is not a mine
        self.grid = np.zeros((self.r, self.c), dtype = int)

        self.generateMines()

    def getNeighbor(self, rowin, colin):            
        COLS = self.c
        ROWS = self.r
        neighbors = []
        for row in range(rowin-1, rowin+2):
            for col in range(colin-1, colin+2):
                if (-1 < rowin < ROWS and 
                    -1 < colin < COLS and 
                    (rowin != row or colin != col) and
                    (0 <= col < COLS) and
                    (0 <= row < ROWS)):
                    neighbors.append((row,col))

        return neighbors
        
    def generateMines(self):
        """
        """
        choices = np.random.choice(self.r * self.c, self.num_of_mines, replace = False)
        cols = np.mod(choices, self.c)
        rows = np.floor_divide(choices, self.c)
        cols = np.reshape(cols, [-1, 1])
        rows = np.reshape(rows, [-1, 1])
        indices = np.concatenate([cols, rows], axis = 1)
        indices.astype(int)
        for r, c in indices:
            self.mines[r][c] = 1
        for i in range(self.r):
            for j in range(self.c):
                if self.mines[i][j] != 1:
                    neighbors = self.getNeighbor(i, j)
                    for r, c in neighbors:
                        if self.mines[r][c] == 1:
                            self.grid[i][j] += 1

    def reveal(self, r, c):
        """
        """

        ## dfs
        visited = self.state_mask + self.mines
        stk = [(r, c)]
        while stk:
            r, c = stk.pop()
            if visited[r][c] != 1:
                visited[r][c] = 1
                self.state_mask[r][c] = 1
                self.num_uncover -= 1

                if self.grid[r][c] == 0:
                    ns = self.getNeighbor(r, c)
                    for nr, nc in ns:
                        stk.append((nr, nc))


    def action_random_true(self):
        """
        """

        visited = self.state_mask + self.mines
        potential_action = []
        for i in range(self.r):
            for j in range(self.c):
                if visited[i][j] == 0:
                    potential_action.append((i, j))

        choice = np.random.choice(len(potential_action), 1)[0]
        choice = potential_action[choice]
        self.reveal(choice[0], choice[1])

        if self.num_uncover == self.num_of_mines: #optimize it!
            return None, None, self.num_of_mines

        state = self.state_mask * self.grid
        ans = self.state_mask + self.mines

        return state, ans, self.num_of_mines


    def action(self, r, c):
        pass

    def reset(self):
        if self.random_assign:
            self.num_of_mines = np.random.randint(math.sqrt(self.r * self.c), math.floor(self.r * self.c / 2) + 1, 1)
        
        self.state_mask = np.zeros((self.r, self.c), dtype = int)
        self.num_uncover = self.r * self.c
        self.mines = np.zeros((self.r, self.c), dtype = int) 
        self.grid = np.zeros((self.r, self.c), dtype = int)

        self.generateMines()