'''
A training engine specifically for Sphero control DQN.
Make data generates data that looks like actual data.
Hard coded to my base variables:
grid size: (640, 480)
    -inputs will always be within this grid
    -actual position allowed a radius r from initial point
    -actual position can be outside the grid
data inputs: 5 coordinate pairs plus the command that generated them
cmd outputs: 4 outputs,
    -0/1 for increase/decrease speed (mod 50 for now- only use lower range of speeds)
    -2/3 for increase/decrease heading (increments of 30 deg for quicker turns)
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

def get_reward(coords, goal = (320, 240), tolerance = 120, mode = 'binary'):
    distance = e_dist(coords, goal)
    if mode == 'binary':
        if distance < tolerance:
            reward = 1
        else:
            reward = 0
    elif mode == 'linear':
        reward = 1 - (distance/tolerance)
    else:
        print("Not yet implemented")
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
        point_reward = get_reward(coords)
        print(coords, point_reward)
        if point_reward < 0:
            r = np.min([int(np.abs(point_reward) * 255), 255])
            g = 0
            b = 0
        elif point_reward > 0:
            r = 0
            g = int(np.abs(point_reward) * 255)
            b = 0
        else:
            r = 0
            g = 0
            b = 255
        color = (r, g, b)
        pygame.draw.circle(screen, color, (x, y), 5)
        pygame.event.get()
        pygame.display.flip()
        time.sleep(0.1)

def play_episode(p=0.1, epsilon=0.5, steps=10, model=None, step_pause = 0.01, frame_count = 5, offset = 0, reward_mode="binary", alpha = 0.05, show = True):
    '''
    generate new training episode
    completely random choices, for now
    p is a control parameter for stochasticity
    '''
    # stochasticity 1: random position and zero direction initialization
    grid_size = (640, 480)
    if show == True:
        pygame.init()
        screen = pygame.display.set_mode(grid_size)
        screen.fill((200, 200, 255))
        pygame.display.flip()
    print("Game offset: ", offset)
    start_coords = random_coords()
    coords = start_coords
    # set up initial observation
    heading = 0
    speed = 0
    last_step = [0]
    for _ in range(frame_count):
        last_step.extend(start_coords)
    last_step = np.array(last_step).reshape(1, 11)
    step_reward = get_reward(start_coords, mode=reward_mode)
    # prepare log
    log = []
    # start playing
    for i in range(steps):
        print("Last reward: ", step_reward)
        if model != None:
            predicts = model.predict(last_step)
            print("Q- predictions: ", predicts[0])
            choice = np.argmax(predicts)
        if (model == None) or (np.random.random() < epsilon):
            choice = randint(0, 3)
        heading = (choice * 90)
        if choice == 0:
            speed += 10 % 50
        elif choice == 1:
            speed -= 10 % 50
        elif choice == 2:
            heading += 30 % 360
        elif choice == 3:
            heading -= 30 % 360
        # each iteration generates one frame for the frame stack.
        # for now hard coding to five, for similar reasoning as our grid size
        frames = []
        rewards = 0
        for _ in range(frame_count):
            theta = (heading + offset) * np.pi/180
            if np.random.random() <= p:
                # stochasticity of sim 2: random variation of direction
                offset += (np.random.random() - 1) * np.pi/3
            delta_x = (speed/frame_count) * np.sin(theta)
            delta_y = (speed/frame_count) * np.cos(theta)
            new_coords = coords.astype(int) + np.array([delta_x, delta_y]).astype(int)
            new_coords = np.minimum(new_coords, grid_size).astype(int)
            new_coords = np.maximum(new_coords, np.zeros(2)).astype(int)
            point_reward = get_reward(new_coords, mode=reward_mode)
            # print("Current reward: ", point_reward)
            rewards += (point_reward)
            frames.extend(new_coords)
            x = new_coords[0]
            y = new_coords[1]
            if point_reward < 0:
                r = np.min([int(np.abs(point_reward) * 255), 255])
                g = 0
                b = 0
            elif point_reward > 0:
                r = 0
                g = int(np.abs(point_reward) * 255)
                b = 0
            else:
                r = 0
                g = 0
                b = 255
            color = (r, g, b)
            # print(x, y)
            # screen.fill((255,255,255))
            if show == True:
                pygame.draw.circle(screen, color, (x, y), 5)
                pygame.display.flip()
                pygame.event.get()
            coords = new_coords
            time.sleep(step_pause)
        values = [choice]
        values.extend(frames)
        values = np.array([values]).reshape(1, 11)
        step_reward = rewards/frame_count
        entry = [last_step, choice, values, step_reward]
        if show == False:
            print("Previous experience: ", last_step)
            print("Choice: ", choice)
            print("Transition: ", values)
            print("Reward: ", step_reward)
        # print(entry)
        if model != None:
            # we now have <s, a, r> and perform the update
            target = predicts
            new_predicts = model.predict(values)
            target[0][choice] = step_reward + alpha * new_predicts[0][choice]
            print("Q-value update: ", target[0])
            # print("Next step predicts: ", new_predicts)
            exp_loss = model.train_on_batch(last_step, target)
            print("Update Loss: ", exp_loss)
        last_step = values
        log.append(entry)
    i = 0
    filestr = "episodes/episode_log_0.p"
    while os.path.isfile(filestr):
        i += 1
        filestr = "episodes/episode_log_" + str(i) + ".p"
    pickle.dump(log, open(filestr, "wb"))
    return log

def experience_replay(model, log, alpha = 0.1):
    loss = 0
    print("Ready to experience replay on log of length: ", len(log))
    for idx in range(len(log)):
        # idx = randint(0, len(log) - 1)
        experience = log[idx]
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Replaying index: ", idx)
        print("S : ", experience[0])
        print("A : ", experience[1])
        print("S': ", experience[2])
        print("R : ", experience[3])
        # we now have <s, a, r> and perform the update
        last_step = experience[0]
        target = model.predict(experience[0])
        print("Q-values: ", target)
        new_predicts = model.predict(experience[2])
        choice = experience[1]
        step_reward = experience[3]
        target[0][choice] = step_reward + alpha * new_predicts[0][choice]
        print("Q update: ", target)
        exp_loss = model.train_on_batch(last_step, target)
        loss += exp_loss
        print("Update Loss: ", exp_loss)
    return loss

def data_gen(p = 1, episodes = 1000, steps = 5):
    for _ in range(episodes):
        play_episode(p, steps)

def train_on_archives(model, folder_path=None, limit = 2, limit_mode = "time"):
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
                log = pickle.load(open(choice, "rb"))
                step_count += len(log[0])
                print("input sample: ", log[0][0])
                print("reward sample: ", log[3][0])
                if model != None:
                    replay_loss = experience_replay(model, log)
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

def baseline_model(optimizer = Adam(), inputs = 11, outputs = 4,
                    layers = [{"size":20,"activation":"tanh"}]):
    num_inputs = inputs
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
    model.add(Dense(num_outputs, activation='linear'))
    model.compile(optimizer = optimizer,
                    loss = "mean_squared_error")
    return model

if __name__ == "__main__":
    layers = [{"size":10,"activation":"tanh"},
              {"size":10,"activation":"tanh"},]
    model = baseline_model(layers = layers)
    # model = None
    # model.load_weights("neokulka_model.h5")
    # replay_log = train_model(model = model, folder_path = None, limit = 10)
    # pickle.dump(open("replay_log.p", "wb"))
    # test_rewards()
    done = "n"
    while done == "n":
        episodes = int(input("How many training episodes?\n"))
        steps = int(input("How many training steps per episode?\n"))
        epsilon_start = min(float(input("Epsilong starting point?\n")), 1)
        show = ("y" == input("Display Training?\n"))
        epsilon_end = epsilon_start/2
        epsilon_delta = (epsilon_start - epsilon_end)/episodes
        logs = []
        for i in range(episodes):
            e = epsilon_start - i*epsilon_delta
            log = play_episode(model = model,
                                steps = steps,
                                epsilon = e,
                                reward_mode = 'linear',
                                show = show)
            logs.append(log)
            if len(logs) == 1:
                r_idx = 0
            else:
                r_idx = randint(0, max(len(logs)-1, 1))
            print("Replaying episode ", r_idx)
            r_log = logs[r_idx]
            replay_loss = experience_replay(model, r_log)
            print("Replay loss: ", replay_loss)
        # train_on_archives(model, limit = 1.66)
        model.save("neokulka_model.h5")
        done = input("Done? (y/n)\n")
    print("Training complete")
