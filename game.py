import numpy as np
import tensorflow as tf
import math


class Game:
    def __init__(self, r = 6, c = 6, num_of_mines = 6, random_assign = False):        
        """
        """
        self.r = r
        self.c = c
        if random_assign:
            self.num_of_mines = np.random.randint(math.sqrt(r * c), math.floor(r * c / 2) + 1, 1)
        self.num_of_mines = num_of_mines
        
        self.state_mask = np.zeros((self.r, self.c), dtype = int)
        self.num_uncover = r * c
        self.mines = None
        self.grid = None

    def getNeighbor(self, r, c):            
        COLS = self.c
        ROWS = self.r
        neighbors = []
        for col in range(colin-1, colin+2):
            for row in range(rowin-1, rowin+2):
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
        choices = np.random.choices(self.r * self.c, self.num_of_mines, replace = False)
        cols = np.reminder(choices, self.c)
        rows = np.floor_divide(choices, self.c)
        cols = np.reshape(cols, [-1, 1])
        rows = np.reshape(rows, [-1, 1])
        indices = np.concatenate([cols, rows], axis = 1)
        indices.astype(int)
        self.mines = tf.sparse.to_dense(tf.sparse.SparseTensor(indices, 1, (self.r, self. c))).numpy()
        self.mines.astype(int)
        for i in range(self.r):
            for j in range(self.c):
                if self.mines[i][j] != 1:
                    neighbors = getNeighbor(i, j)
                    for n in neighbors:
                        

    def reveal(self, r, c):
        """
        """
        


    def action_random_ture(self):
        """
        """

    def action(self, r, c):
        pass
        
