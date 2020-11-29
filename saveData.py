import numpy as np
import math
from game import Game

def main(): 
    # model = Game()
    # a = model.action_random_true()
    # print("a: ")
    # print(a)
    # print()

    # b = model.action_random_true()
    # print("b: ")
    # print(b)
    # print()

    # c = np.array(a+b, dtype = object)

    # np.save("filename.npy",c)

    # loaded = np.load("filename.npy", allow_pickle=True)
    # print(loaded)

    model = Game()
    mylist = []
    num = 10000
    i = 0
    n_game = 1
    while(i < num):
        a = model.action_random_true()
        if(a[0] is None and a[1] is None):
            model.reset()
            n_game+=1
        else:
            mylist.append(a)
            i+=1
    mat = np.array(mylist, dtype = object)
    np.save("data10.npy",mat)
    loaded = np.load("data10.npy", allow_pickle=True)
    print(loaded)
    print("n_game: ",n_game)


if __name__ == '__main__':
	main()
