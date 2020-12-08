import numpy as np
import math
import tensorflow as tf
from model import Model
from game import Game
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



def train(model, train_state, train_mines, train_answer, first):
    epoch_num = 3
    train_state = tf.Variable(train_state)
    train_mines = tf.Variable(train_mines)
    train_answer = tf.Variable(train_answer)
    first = tf.Variable(first)
    size = train_state.shape[0]

    final_loss=[]
    final_accuracy=[]

    for epoch in range(epoch_num):
        shuffle_index = np.arange(0, train_state.shape[0])
        np.random.shuffle(shuffle_index)
        train_state = tf.gather(train_state, shuffle_index)
        train_mines = tf.gather(train_mines, shuffle_index)
        train_answer = tf.gather(train_answer, shuffle_index)
        first = tf.gather(first, shuffle_index)
        #train_state, train_mines, train_answer = train_state[shuffle_index], train_mines[shuffle_index], train_answer[shuffle_index]
        batch_size = 256
        iteration = 0
        for i in range(0, size, batch_size):
            end = min(i + batch_size, size)
            sample_state = train_state[i: end]
            sample_mines = train_mines[i: end]
            sample_answer = train_answer[i: end]
            sample_first = first[i: end]
            #sample_state, sample_mines, sample_answer = train_state[i * batch_size: i * batch_size + batch_size], train_mines[i * batch_size: i * batch_size + batch_size], train_answer[i * batch_size: i * batch_size + batch_size]

            grids = sample_mines / 36
            grids = tf.reshape(grids, shape=(grids.shape[0], 1, 1, 1))
            grids = tf.repeat(grids, 6, axis=1)
            grids = tf.repeat(grids, 6, axis=2)
            # extra_features = tf.expand_dims(tf.argmax(sample_state, axis=3), axis=3)
            # extra_features = tf.cast(extra_features, tf.float32)
            grids = tf.cast(grids, tf.float32)
            # extra_features = tf.concat((grids, extra_features), axis=3)
            #extra_features = tf.repeat(extra_features, repeats=32, axis=3)


            extra_features = tf.concat((grids, sample_first), axis = -1)

            with tf.GradientTape() as tape:
                output = model(sample_state, extra_features)
                loss, _, accuracy = model.loss(output, sample_answer)
                loss = tf.reduce_mean(loss)

                final_loss.append(loss)
                final_accuracy.append(accuracy)
                if i > 0 and i % (20 * 256) == 0:
                    print("Epoch: {} Iteration: {}/{} Loss: {} accuracy: {}".format(epoch, iteration,
                                                                                    size // batch_size,
                                                                                    loss, accuracy))
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            iteration += 1
    visualize_loss(final_loss)
    visualize_accuracy(final_accuracy)

def visualize_loss(losses): 
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig("Loss.png")

def visualize_accuracy(accuracies): 
    x = [i for i in range(len(accuracies))]
    plt.plot(x, accuracies)
    plt.title('Accuracy per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.savefig("Accuracy.png")

def test(model):
    g = Game(num_of_mines = 6, random_assign = False)
    wins = 0
    num_of_mines = 6
    grids = float(num_of_mines) / 36
    grids = tf.reshape(grids, shape=(1, 1, 1, 1))
    grids = tf.repeat(grids, 6, axis=1)
    grids = tf.repeat(grids, 6, axis=2)
    for i in range(10000):
        flag = True
        num_of_mines = 6

        state, _, num_of_mines = g.action_random_true()

        test_v = 0
        while(state is None):
            g.reset()
            state, _, num_of_mines = g.action_random_true()
            if(test_v > 100):
                print("there are error")
                break
            test_v += 1
        

        state = tf.Variable(state)
        state = tf.expand_dims(state, axis = 0)

        first = state % 9 
        first = tf.cast(first, tf.float32)
        first = tf.expand_dims(first, axis = -1)

        state = tf.one_hot(state, 10)

        extra_features = tf.concat((grids, first), axis = -1)

        logits = model(state, extra_features)

        output_argmax = tf.argmax(logits, axis=1)

        r = math.floor(output_argmax[0] / 6)
        c = int(output_argmax[0] % 6)

        times = 0

        while flag:
            success, state, end_of_game = g.action(r, c)
            times += 1
            if times > 36:
                print("problems in test")
                print(state)
                break
            if not success:
                break
            elif success and end_of_game:
                wins += 1
                break
            else:
                state = tf.Variable(state)
                state = tf.expand_dims(state, axis = 0)
                first = state % 9 
                first = tf.cast(first, tf.float32)
                first = tf.expand_dims(first, axis = -1)
                state = tf.one_hot(state, 10)
                extra_features = tf.concat((grids, first), axis = -1)
                logits = model(state, extra_features)
                output_argmax = tf.argmax(logits, axis=1)
                r = math.floor(output_argmax[0] / 6)
                c = int(output_argmax[0] % 6) 
                
        g.reset() 

    return float(wins) / 10000.0


def main():
    loaded = np.load("data.npy", allow_pickle=True)
    np.random.shuffle(loaded)
    states = [m[0] for m in loaded]
    answers = [m[1] for m in loaded]
    nums_of_mines = [float(m[2]) for m in loaded]
    states = np.array(states)
    answers = np.array(answers)
    flip = np.ones(answers.shape)
    answers = flip - answers
    nums_of_mines = np.array(nums_of_mines)
    first = states % 9
    first = tf.cast(first, tf.float32)
    first = tf.expand_dims(first, axis = -1)
    states = tf.one_hot(states, 10)
    # states_new = np.zeros((states.size, 10))
    # states_flatten = states.flatten()
    # for i, j in enumerate(states_flatten):
    #     states_new[i][j] = 1
    # states = np.reshape(states_new, newshape=(states.shape[0], states.shape[1], states.shape[2], -1))
    # training_data = len(states) * 0.9
    # training_data = int(training_data)
    # train_state = states[:training_data]
    # test_state = states[training_data:]
    # train_answer = answers[:training_data]
    # test_answer = answers[training_data:]
    # train_mines = nums_of_mines[:training_data]
    # test_mines = nums_of_mines[training_data:]
    model = Model(width=6, height=6)
    train(model=model, train_state=states, train_answer=answers, train_mines=nums_of_mines, first = first)
    acc = test(model)

    print(acc)

if __name__ == '__main__':
    main()
