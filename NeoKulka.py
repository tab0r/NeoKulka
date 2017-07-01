'''
A training engine specifically for Sphero control DQN.
Make data generates data that looks like actual data.
Hard coded to my base variables:
grid size: (640, 480)
    -inputs will always be within this grid
    -actual position allowed a radius r from initial point
    -actual position can be outside the grid
data inputs: 5 coordinate pairs plus the command that generated them
cmd outputs: 9 outputs,
    -8 for stay
    -0-3 for low speed at 90 degree angles
    -4-7 for high speed at 90 degree angles
'''
import time, os, sys, pickle, pygame, pdb
import numpy as np
from scipy.spatial.distance import euclidean as e_dist
from os import listdir
from os.path import isfile, join
from random import randint

# neural network imports
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.optimizers import Adam

def random_coords(grid_size = (640, 480)):
    return np.array([randint(0, grid_size[0]-1), randint(0, grid_size[1]-1)])

def get_reward(coords, goal = (320, 240), tolerance = 240):
    distance = e_dist(coords, goal)
    reward = 1 - (distance/tolerance)
    return reward

def test_rewards():
    grid_size = (640, 480)
    pygame.init()
    screen = pygame.display.set_mode(grid_size)
    screen.fill((0, 0, 0))
    pygame.display.flip()
    for _ in range(100):
        coords = random_coords()
        x = coords[0]
        y = coords[1]
        print(coords, get_reward(coords))
        color = (0, 128, 255)
        pygame.draw.circle(screen, color, (x, y), 5)
        pygame.display.flip()
        time.sleep(0.1)

def play_episode(p = 0.1, steps = 10, model = None):
    # eventually allow agent to go off-screen. make it work with handicap first
    # max_r = 240):
    '''
    generate new training episode
    completely random choices, for now
    p is a control parameter for stochasticity
    '''
    # once this works properly, this offset will make the sim better reflect the conditions I expect the DQN to encounter- specifically, in each "game", the Sphero will have a random zero heading, and that heading will change over time due to errors.
    # stochasticity 1: random position and zero direction initialization
    grid_size = (640, 480)
    pygame.init()
    screen = pygame.display.set_mode(grid_size)
    screen.fill((0, 0, 0))
    pygame.display.flip()
    offset = randint(0, 359)
    print("Game offset: ", offset)
    start_coords = random_coords()
    coords = start_coords
    # set up initial observation
    frame_count = 5
    last_step = [9]
    for _ in range(frame_count):
        last_step.extend(start_coords)
    last_step = np.array(last_step).reshape(1, 11)
    # prepare log
    log = []
    # start playing
    for i in range(steps):
        if model != None:
            predicts = model.predict(last_step)
            choice = np.argmax(predicts)
        elif (model == None) or (np.random.random() < 0.3):
            choice = randint(0, 3)
        heading = (choice * 90)
        # once the network is training well, I'll either make it bigger  to control speed, too, or train another network specifically for controlling speed.
        # speed = randint(0, 10) * 25
        if choice != 4:
            speed = 100
        else:
            speed = 0
        # each iteration generates one frame for the frame stack.
        # for now hard coding to five, for similar reasoning as our grid size
        frames = []
        rewards = 0
        for _ in range(frame_count):
            theta = (heading + offset) * np.pi/180
            if np.random.random() <= p:
                # stochasticity of sim 2: random variation of direction
                theta += (np.random.random() - 1) * np.pi/5
            delta_x = (speed/frame_count) * np.sin(theta)
            delta_y = (speed/frame_count) * np.cos(theta)
            new_coords = coords.astype(int) + np.array([delta_x, delta_y]).astype(int)
            new_coords = np.minimum(new_coords, grid_size).astype(int)
            new_coords = np.maximum(new_coords, np.zeros(2)).astype(int)
            rewards += (get_reward(new_coords))
            frames.extend(new_coords)
            x = new_coords[0]
            y = new_coords[1]
            color = (255, 255, 255)
            # print(x, y)
            # screen.fill((255,255,255))
            pygame.draw.circle(screen, color, (x, y), 5)
            pygame.display.flip()
            coords = new_coords
            time.sleep(0.1)
        values = [heading]
        values.extend(frames)
        values = np.array([values]).reshape(1, 11)
        step_reward = rewards/frame_count
        print(values, step_reward)
        log.append(values)
        if model != None:
            # we now have <s, a, r> and perform the update
            target = predicts
            new_predicts = model.predict(values)
            target[0][choice] = step_reward + new_predicts[0][choice]
            model.train_on_batch(last_step, target)
        last_step = values
    return log

def data_gen(p = 1, episodes = 1000, steps = 5):
    for _ in range(episodes):
        play_episode(p, steps)

def train_model(model, folder_path, limit = 10, limit_mode = "time"):
    '''
    train the model on data in folder_path
    assumes files are episodes of training
    randomly selects until limit reached
    by default, limits training time to ten minutes
    '''
    losses = []
    if limit_mode == "time":
        t0 = time.time()
        t1 = time.time()
        t_str = "Training for " + str(limit) + " minutes."
        step_count = 0
        count = 0
        print(t_str)
        while((t1 - t0) < (limit * 60)):
            count += 1
            # print("t1-t0:", t1-t0)
            if folder_path == None:
                mypath = "episodes/"
            else:
                mypath = folder_path
            filenames = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
            choice = filenames[randint(0, len(filenames) - 1)]
            if choice == "episodes/.DS_Store":
                pass
            else:
                print("Filename: ", choice)
                output = pickle.load(open(choice, "rb"))
                step_count += len(output[0])
                print("input sample: ", output[0][0])
                print("target sample: ", output[1][0])
                if model != None:
                    replay_loss = model.train_on_batch(output[0], output[1])
                    losses.append(replay_loss)
                    model_filestr = "offline_training_backup.h5"
                    model.save(model_filestr)
                    print("loss on replay training: ", replay_loss)
                # time.sleep(1)
                t1 = time.time()
        end_str = "Complete. Saw " + str(step_count) + " steps in " + str(count) \
                    + " episodes."
        print(end_str)
        print("Execution time: ", t1-t0)
    else:
        print("Not yet implemented")
    return losses

def baseline_model(optimizer = Adam(), inputs = 11, outputs = 5,
                    layers = [{"size":20,"activation":"relu"}]):
    # two inputs - each coordinate
    num_inputs = inputs
    # four outputs - one for each potential offset
    num_outputs = outputs
    # prepare the navigator model
    model = Sequential()
    # initial inputs
    l = list(layers)
    l0 = l[0]
    del l[0]
    model.add(Dense(l0['size'],
                    input_dim = num_inputs,
                    activation = l0['activation']))
    # the hidden layers
    for layer in l:
        model.add(Dense(layer['size'], activation=layer['activation']))
    # the output layer
    model.add(Dense(num_outputs, activation='tanh'))
    model.compile(optimizer = optimizer,
                    loss = "mean_squared_error")
    return model

if __name__ == "__main__":
    model = baseline_model()
    # model = None
    # model.load_weights("sphero_model.h5")
    # replay_log = train_model(model = model, folder_path = None, limit = 10)
    # model.save("sphero_model.h5")
    # pickle.dump(open("replay_log.p", "wb"))
    # test_rewards()
    log = play_episode(model = model)
