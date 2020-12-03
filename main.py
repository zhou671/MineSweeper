import numpy as np
import math
import tensorflow as tf
from model import Model


def train(model, train_state, train_mines, train_answer):
    epoch_num = 5
    for epoch in range(epoch_num):
        shuffle_index = np.arange(0, len(train_state))
        np.random.shuffle(shuffle_index)
        train_state, train_mines, train_answer = train_state[shuffle_index], train_mines[shuffle_index], train_answer[
            shuffle_index]
        batch_size = 32
        for i in range(0, len(train_state) // batch_size):
            sample_state, sample_mines, sample_answer = train_state[
                                                        i * batch_size: i * batch_size + batch_size], \
                                                        train_mines[
                                                        i * batch_size: i * batch_size + batch_size], train_answer[
                                                                                                      i * batch_size: i * batch_size + batch_size]
            sample_state = tf.convert_to_tensor(sample_state)
            sample_mines = tf.convert_to_tensor(sample_mines)
            sample_answer = tf.convert_to_tensor(sample_answer)
            grids = sample_mines / 36
            grids = tf.reshape(grids, shape=(grids.shape[0], 1, 1, 1))
            grids = tf.repeat(grids, 6, axis=1)
            grids = tf.repeat(grids, 6, axis=2)
            extra_features = tf.expand_dims(tf.argmax(sample_state, axis=3), axis=3)
            extra_features = tf.cast(extra_features, tf.float32)
            grids = tf.cast(grids, tf.float32)
            extra_features = tf.concat((grids, extra_features), axis=3)
            extra_features = tf.repeat(extra_features, repeats=32, axis=3)
            with tf.GradientTape() as tape:
                output = model(sample_state, extra_features)
                loss, _, accuracy = model.loss(output, sample_answer)
                loss = tf.reduce_mean(loss)
                if i > 0 and i % 20 == 0:
                    print("Epoch: {} Iteration: {}/{} Loss: {} accuracy: {}".format(epoch, i,
                                                                                    len(train_state) // batch_size,
                                                                                    loss, accuracy))
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_state, test_mines, test_answer):
    shuffle_index = np.arange(0, len(test_state))
    np.random.shuffle(shuffle_index)
    train_state, train_mines, train_answer = test_state[shuffle_index], test_mines[shuffle_index], test_answer[
        shuffle_index]
    batch_size = 32
    total_ac = []
    for i in range(0, len(train_state) // batch_size):
        sample_state, sample_mines, sample_answer = train_state[
                                                    i * batch_size: i * batch_size + batch_size], train_mines[
                                                                                                  i * batch_size: i * batch_size + batch_size], train_answer[
                                                                                                                                                i * batch_size: i * batch_size + batch_size]
        sample_state = tf.convert_to_tensor(sample_state)
        sample_mines = tf.convert_to_tensor(sample_mines)
        sample_answer = tf.convert_to_tensor(sample_answer)
        grids = sample_mines / 36
        grids = tf.reshape(grids, shape=(grids.shape[0], 1, 1, 1))
        grids = tf.repeat(grids, 6, axis=1)
        grids = tf.repeat(grids, 6, axis=2)
        extra_features = tf.expand_dims(tf.argmax(sample_state, axis=3), axis=3)
        extra_features = tf.cast(extra_features, tf.float32)
        grids = tf.cast(grids, tf.float32)
        extra_features = tf.concat((grids, extra_features), axis=3)
        extra_features = tf.repeat(extra_features, repeats=32, axis=3)
        output = model(sample_state, extra_features)
        output_argmax = tf.argmax(output, axis=1)
        sample_answer = tf.reshape(sample_answer, shape=(sample_answer.shape[0], -1))
        pre_label = tf.gather(sample_answer, output_argmax, axis=1)
        ac = tf.reduce_mean(tf.cast(pre_label, tf.float32))
        total_ac.append(ac)
    print("Total accuracy: {}".format(tf.reduce_mean(total_ac)))
    return tf.reduce_mean(total_ac)


def main():
    loaded = np.load("data.npy", allow_pickle=True)
    np.random.shuffle(loaded)
    states = [m[0] for m in loaded]
    answers = [m[1] for m in loaded]
    nums_of_mines = [float(m[2]) for m in loaded]
    states = np.array(states)
    answers = np.array(answers)
    nums_of_mines = np.array(nums_of_mines)
    states_new = np.zeros((states.size, 10))
    states_flatten = states.flatten()
    for i, j in enumerate(states_flatten):
        states_new[i][j] = 1
    states = np.reshape(states_new, newshape=(states.shape[0], states.shape[1], states.shape[2], -1))
    training_data = len(states) * 0.9
    training_data = int(training_data)
    train_state = states[:training_data]
    test_state = states[training_data:]
    train_answer = answers[:training_data]
    test_answer = answers[training_data:]
    train_mines = nums_of_mines[:training_data]
    test_mines = nums_of_mines[training_data:]
    model = Model(width=6, height=6)
    train(model=model, train_state=train_state, train_answer=train_answer, train_mines=train_mines)
    test(model=model, test_state=test_state, test_answer=test_answer, test_mines=test_mines)


if __name__ == '__main__':
    main()
