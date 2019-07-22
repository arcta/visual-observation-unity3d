import gym
import numpy as np
import numpy.random as rnd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sb

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Input, Dense, Flatten, Concatenate, Reshape
from keras.layers import BatchNormalization, Dropout
from keras.layers import Activation, LeakyReLU


IMG_DIM = (64, 64, 3) # unity default is (84, 84, 3)
ACT_DIM = (2,) # azimuth, altitude
N_CHANNELS = 3

# agent's `brain capacity`
LATENT_DIM = 8
MEMORY_LENGTH = 4

# agent's `cortex structure`
N_FILTERS = 16
N_LAYERS = 3
OPTIMIZER = 'adam'

# environment directions
ACTION = {
    'Right':[1., 0.],
    'Left':[-1., 0.],
    'Up':[0., -1.],
    'Down':[0., 1.],
    'View':[0., 0.] }


# collect observations executing squense of actions
def run_seq(env, actions, start = np.zeros((2,))):
    x = start
    observations, coords = [],[]
    for action in actions:
        view, _,__,___ = env.step(action)
        observations.append(view)
        x += action
        coords.append(tuple(x))
    return observations, coords


# show observed sequence
def show_seq(actions, observations, coords):
    n = len(observations)
    fig, ax = plt.subplots(1, n, figsize = (16,3))
    for i in range(n):    
        ax[i].imshow(observations[i])
        ax[i].set_title('A[%.0f, %.0f]' % tuple(actions[i]))
        ax[i].set_xlabel('X[%.0f, %.0f]' % tuple(coords[i]))
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.show()


# get a continuous range of coords covered in defined #steps
def smooth_trajectory(steps):
    def trajectory(x):
        x += 45 # move down to set `cumulative zero` at zero
        return np.pi * np.sin(2 * x * np.pi / steps)
    return trajectory


AMPL = np.array([5., 1.])
STEPS = np.array([179., 83.])
START = np.array([0., 37.])


def init_trajectory(env, start = START, steps = STEPS):
    env.reset()
    _,__,__,___ = env.step(start)
    return smooth_trajectory(steps)


def make_move(env, trajectory, index, ampl = AMPL):
    action = ampl * trajectory(index)
    observation, _,__,___ = env.step(action)
    return observation, action


# common training rutine
def train_online(env, model, make_batch, epochs = 100, batch_size = 128,
                 shuffle = True, verbose = 0, visual = 0):
    # performance tracking
    log = {}
    # agent path (Trajectory.ipynb)
    path = init_trajectory(env)
    # running coordinates
    x = START.copy()
    # step counter
    index = 0
    # run: online epoch here means collecting 2 batches of data
    for e in range(epochs):
        # initialize 2 batches (train + test) to hold observations, actions and coords
        V = np.empty([batch_size * 2] + list(IMG_DIM))
        A = np.empty([batch_size * 2] + list(ACT_DIM))
        X = np.empty([batch_size * 2] + list(ACT_DIM))
        # collect a batch of data walking around
        for j in range(2 * batch_size):
            observation, action = make_move(env, path, index)
            V[j,:] = observation
            A[j,:] = action
            index += 1
            x += action
            X[j,:] = x
        if model and make_batch:
            # build inputs/targets
            data, labels = make_batch(V, A, X)
            # multiple inputs
            if isinstance(data, list):
                b = data[0].shape[0]//2
                inputs = [[d[:b,:] for d in data], [d[b:,:] for d in data]]
            else:
                b = data.shape[0]//2
                inputs = [data[:b,:], data[b:,:]]
            # multiple outputs
            if isinstance(labels, list):
                targets = [[d[:b,:] for d in labels], [d[b:,:] for d in labels]]
            else:
                targets = [labels[:b,:], labels[b:,:]]
            # train the model using first collected batch, test using second
            track = model.fit(inputs[0], targets[0], b, validation_data = (inputs[1], targets[1]),
                                  epochs = 2, shuffle = shuffle, verbose = verbose)
            # log performance
            for key in track.history:
                log[key] = log.get(key, []) + [np.mean(track.history[key])]
            # switch train and test data    
            track = model.fit(inputs[1], targets[1], b, validation_data = (inputs[0], targets[0]),
                                  epochs = 2, shuffle = shuffle, verbose = verbose)
            for key in track.history:
                log[key][-1] = (log[key][-1] + np.mean(track.history[key]))/2.
        # print progress
        if e == 0 or (e + 1)%(epochs//10) == 0:
            if visual:
                d = min(12, 2 * batch_size)
                show_seq(A[-d:,:], V[-d:,:], X[-d:,:])
            if 'loss' in log and 'val_loss' in log:
                print('{:>7} Loss  trainig: {:.4f} validation: {:.4f}'\
                        .format(e + 1, log['loss'][-1], log['val_loss'][-1]))
    # return log and last epoch data
    return log, V, A, X


# plot training history: train vs. validation loss
def plot_history(log, title = 'Loss history', offset = 0):
    models = ['_'.join(x.split('_')[1:-1]) for x in log.keys() if x[:3] == 'val' and len(x.split('_')) > 2]
    if len(models) == 0: models = ['loss']
    N, n, k = len(log['loss']), len(log['val_loss'])//10, len(models)
    fig, ax = plt.subplots(len(models), figsize = (8, 3 * k), sharex = True)
    for i,m in enumerate(models):
        a = ax if k == 1 else ax[i]
        s = m if m == 'loss' else m +'_loss'
        x = [i for i in range(offset, N)]
        a.plot(x, log['val_'+ s][offset:], color = 'C1', label = 'Validation')
        a.plot(x, log[s][offset:], color = 'C0', label = 'Training', alpha = .6)
        a.hlines(y = np.mean(log['val_'+ s][-n:]), xmin = offset, xmax = N, linestyle = ':')
        a.set_title(title if m == 'loss' else m)
        a.legend()
    plt.show()


# compare visual observation and predicted view
def plot_predicted_vs_observed(views, actions, predictions, observations):
    k = len(views)
    fig, ax = plt.subplots(3, k, figsize = (15, 6))
    for i in range(k):
        ax[0][i].set_title('Current')
        ax[0][i].imshow(views[i])
        ax[1][i].imshow(predictions[i])
        ax[2][i].imshow(observations[i])
        for j,t in enumerate(['Move [%.0f, %.0f]' % tuple(actions[i]), 'Predicted', 'Observed']):
            ax[j][i].set_xlabel(t)
            ax[j][i].set_xticks([])
            ax[j][i].set_yticks([])
    plt.show()


# value relevance distribution
def encoding_heatmap(env, model, latent_dim, cmap = 'Reds'):
    diff = np.zeros((5, latent_dim))
    ampl = 10.
    env.reset()
    for i,action in enumerate(ACTION.values()):
        action = np.array(action)
        for _ in range(100):
            loc = [rnd.randint(-350, 350), rnd.randint(-50, 50)] 
            o1, _,__,___ = env.step(loc)
            e1 = model.predict([np.array([o1]), np.array([[0., 0.]])])
            o2, _,__,___ = env.step(action * ampl)
            e2 = model.predict([np.array([o2]), np.array([action])/360.])
            diff[i,:] += ((e1 - e2) ** 2).reshape(-1)
            e1 = e2
    fig, ax = plt.subplots(figsize = (12, 3))
    sb.heatmap(np.sqrt(diff)/100, ax = ax, yticklabels = list(ACTION.keys()), cmap = cmap)
    plt.title('Encoding: action-value heatmap')
    plt.yticks(rotation = 0)
    plt.show()


