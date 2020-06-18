from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
from scipy.spatial import distance
from point import Point
import random
import math
import pygame


WIDTH = 800
HEIGHT = 600
PLAYER_SIZE = 45
BALL_SIZE = 15

class ReplayBuffer(object):
    def __init__(self,mem_size, input_shape, n_actions, discrete=False):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        # store one hot encoding of actions, if appropriate
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = Sequential([
                Dense(fc1_dims, input_shape=(input_dims,)),
                Activation('tanh'),
                Dense(fc2_dims),
                Activation('tanh'),
                Dense(fc2_dims),
                Activation('tanh'),
                Dense(n_actions)])

    model.compile(optimizer=Adam(lr=lr), loss='mse')

    return model

class Player(object):
    def __init__(self, pitch, team='team1', angle=90):

        self.team=team
        self.pitch = pitch

        self.position = Point(random.randrange(350, 450), random.randrange(250, 350))
        self.reward = 0
        self.angle = angle
        self.n_actions = 9
        self.action_space = [0,1,2,3,4,5,6,7,8]
        self.model_name=f"actor{self.team}.h5"
        self.gamma=0.95
        self.alpha = 0.005
        self.beta = 0.005
        self.input_dims = 12
        self.fc1_dims = 16
        self.fc2_dims = 8
        self.epsilon = 1
        self.epsilon_dec = 0.9999991
        self.epsilon_min = 0.05
        self.batch_size = 16
        self.model_file = 'ddqn.h5'
        self.replace_target = 1000
        self.mem_size= 1_0000
        self.memory = ReplayBuffer(self.mem_size, self.input_dims, self.n_actions,
                                   discrete=True)
        self.q_eval = build_dqn(self.alpha, self.n_actions, self.input_dims, 256, 256)
        self.q_target = build_dqn(self.alpha, self.n_actions, self.input_dims, 256, 256)
        if self.team =='team1':
            self.position =  Point(random.randrange(350, 450), random.randrange(250, 350))
            self.image = pygame.image.load("kit.png")
        else:
            self.x = Point(random.randrange(WIDTH/2+30, WIDTH-30), random.randrange(20, HEIGHT))
            self.image = pygame.image.load("kit2.png")
        self.image = pygame.transform.scale(self.image, (PLAYER_SIZE, PLAYER_SIZE))

    def return_rotation(self):
        return pygame.transform.rotate(self.image, self.angle)

    def reset_position(self):
        self.position = Point(random.randrange(350, 450), random.randrange(250, 350))


    def move(self,speed=3):

        if self.position.x <self.pitch.lower_x_pitch:
            self.position.x = self.pitch.lower_x_pitch
        if self.position.x > self.pitch.upper_x_pitch:
            self.position.x = self.pitch.upper_x_pitch
        if self.position.y < self.pitch.lower_y_pitch:
            self.position.y = self.pitch.lower_y_pitch
        if self.position.y > self.pitch.upper_y_pitch:
            self.position.y =self.pitch.upper_y_pitch
        self.getNewPos(speed)

    def rotate(self, direction=1):
        self.angle =  self.angle + 20*(direction*2-3)
        if self.angle>360:
            self.angle=0
        if self.angle<0:
            self.angle=360

    def getNewPos(self, speed):

        old_x, old_y = self.position.x, self.position.y
        angle = float(self.angle)
        delta_y = speed*2 * math.cos(math.radians(angle))
        delta_x = speed*2 * math.sin(math.radians(angle))
        self.position = Point(self.position.x - delta_x, self.position.y - delta_y)

    def kick_ball(self, ball, power=6):

        if self.position.distance_between_points(ball.position)<20 and ball.speed==0:
            ball.player=self
            ball.is_kicked=True
            ball.player_angle = self.angle + random.randint(-20,20)
            ball.speed=20
            ball.angle = self.angle
            ball.power =power

    def return_observation(self,opponent, ball):

        # return np.array([self.position.x/WIDTH,self.position.y/HEIGHT, math.radians(self.angle)/math.pi,  ball.position.x/WIDTH, ball.position.y/HEIGHT, self.position.distance_between_points(ball.position)/WIDTH, self.position.angle_between_points(ball.position)/2/math.pi])
        return np.array([math.radians(self.angle)/math.pi, self.position.distance_between_points(ball.position)/WIDTH, self.position.angle_between_points(ball.position)/2/math.pi, 
        self.position.x/WIDTH, ball.position.x/WIDTH, ball.position.y/HEIGHT, self.position.y/HEIGHT, opponent.position.x/WIDTH, opponent.position.y/HEIGHT, opponent.position.distance_between_points(ball.position)/WIDTH,
        opponent.position.distance_between_points(self.position)/WIDTH, self.position.angle_between_points(opponent.position)/2/math.pi])

    def pass_ball(self):
        pass
    def shoot_ball(self):
        pass


    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state, ball, goal1, goal2, opponent):
        ball.is_kicked = False
        state = state[np.newaxis, :]
        rand = np.random.random()

        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
        if self.position.distance_between_points(ball.position)<20:
            self.kick_ball(ball,power=action)
            self.reward = 0.1
        if goal1==True:
            if self.team=='team1':
                self.reward=-1
                opponent.reward=1
            else:
                self.reward= 1
                opponent = -1
        elif goal2==True:
            if self.team=='team1':
                self.reward=1
                opponent.reward=-1
            else:
                self.reward=-1
                opponent.reward = 1
        else:
            old_position = self.position.distance_between_points(ball.position)
            if action <3:
                self.rotate(direction=action)
            elif action <5:
                self.last_position = self.position
                self.move(speed=action)
            self.reward=(old_position-self.position.distance_between_points(ball.position))/WIDTH
        return action

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = \
                                          self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            q_next = self.q_target.predict(new_state)
            q_eval = self.q_eval.predict(new_state)
            q_pred = self.q_eval.predict(state)

            max_actions = np.argmax(q_eval, axis=1)

            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + \
                    self.gamma*q_next[batch_index, max_actions.astype(int)]*done

            _ = self.q_eval.fit(state, q_target, verbose=0)

            self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
                           self.epsilon_min else self.epsilon_min
            if self.memory.mem_cntr % self.replace_target == 0:
                self.update_network_parameters()

    def update_network_parameters(self):
        self.q_target.model.set_weights(self.q_eval.model.get_weights())

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
        # if we are in evaluation mode we want to use the best weights for
        # q_target
        if self.epsilon == 0.0:
            self.update_network_parameters()


class Goalkeeper(Player):
  def __init__(self,pitch, team='team1', angle=90):
    super().__init__(pitch, team, angle)

class Defender(Player):
    def __init__(self,pitch, team='team1',angle=90):
        super().__init__(pitch,team, angle)

class Midfielder(Player):
    def __init__(self,pitch, team='team1', angle=90):
        super().__init__(pitch, team, angle)

class Striker(Player):
    def __init__(self,pitch, team='team1', angle=90):
        super().__init__(pitch, team, angle)
