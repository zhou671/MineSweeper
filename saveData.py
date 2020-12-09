import numpy as np
import random
import math
from game import Game

def main(): 
    model = Game()
    mylist = []
    num = 1000000
    i = 0
    n_game = 1
    random_array = [2,3,4]
    r_a = np.random.choice(random_array, 1, p=[0.30, 0.40, 0.30])
    num_random = r_a[0]
    list_element = []
    while(i < num):
        a = model.action_random_true()
        if(a[0] is None and a[1] is None):
            model.reset()
            r_a = np.random.choice(random_array, 1, p=[0.30, 0.40, 0.30])
            num_random = r_a[0]
            n_game+=1
            if(num_random >= num - i):
                # print("num_random: ", num_random)
                # print("num - i: ", num - i)
                # print("len(mylist): ", len(mylist))
                # print("len(list_element): ", len(list_element))
                # print("list_element: ", list_element[:(num - i)])
                mylist.extend(list_element[:(num - i)])
            else:
                if(len(list_element) < num_random):
                    mylist.extend(list_element)
                else:
                    a2 = random.sample(list_element, num_random)
                    mylist.extend(a2)
                list_element = []
            i = len(mylist)
        else:
            list_element.append(a)

    mat = np.array(mylist, dtype = object)
    np.save("data.npy",mat)

    loaded = np.load("data.npy", allow_pickle=True)
    print(loaded)
    print(loaded.shape)
    print("n_game: ",n_game)


if __name__ == '__main__':
	main()
