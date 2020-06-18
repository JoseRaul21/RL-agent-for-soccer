import math
import random
import pygame
from scipy.spatial import distance
from point import Point
import numpy as np
from keras import backend as K 
from keras.layers import Activation, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Activation
from keras.models import load_model
from keras import Sequential

WIDTH = 800
HEIGHT = 600
PLAYER_SIZE = 35
BALL_SIZE = 15
class Player:
    def __init__(self, pitch, team='team1', angle=90):

        self.team=team
        self.pitch = pitch

        self.position = Point(random.randrange(350, 450), random.randrange(250, 350))
        self.last_position = self.position
        self.angle = angle
        self.n_actions = 8
        self.action_space = [0,1,2,3,4,5,6,7]
        self.model_name="actor.h5"
        self.gamma=0.99
        self.alpha = 0.001
        self.beta = 0.001
        self.input_dims = 6
        self.fc1_dims = 256
        self.fc2_dims = 128
        self.reward = 0

        self.actor, self.critic, self.policy = self.build_actor_critic_network()
        
        if self.team =='team1':
            self.position =  Point(random.randrange(350, 450), random.randrange(250, 350))
            self.image = pygame.image.load("kit.png")
        else:
            self.x = Point(random.randrange(WIDTH/2+30, WIDTH-30), random.randrange(20, HEIGHT))
            self.image = pygame.image.load("kit2.png")
        self.image = pygame.transform.scale(self.image, (PLAYER_SIZE, PLAYER_SIZE))


    def return_rotation(self):
        rotated_image = pygame.transform.rotate(self.image, self.angle)
        return rotated_image

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
        delta_y = speed * math.cos(math.radians(angle))
        delta_x = speed * math.sin(math.radians(angle))
        self.position = Point(self.position.x - delta_x, self.position.y - delta_y)

    def kick_ball(self, ball, power=6):

        if self.position.distance_between_points(ball.position)<15 and ball.speed==0:
            ball.player=self
            ball.is_kicked=True
            ball.player_angle = self.angle + random.randint(-20,20)
            ball.speed=20
            ball.angle = self.angle
            ball.power =power

    def return_observation(self, ball):

        return np.array([self.position.x/WIDTH,self.position.y/HEIGHT, ball.position.x/WIDTH, ball.position.y/HEIGHT, self.position.distance_between_points(ball.position)/WIDTH, self.position.angle_between_points(ball.position)/2/math.pi])

    def pass_ball(self):
        pass
    def shoot_ball(self):
        pass

        
    def learn(self, state, action, reward, state_, done):
        state = state[np.newaxis,:]
        state_ = state_[np.newaxis,:]
        critic_value_ = self.critic.predict(state_)
        critic_value = self.critic.predict(state)

        target = reward + self.gamma*critic_value_*(1-int(done))
        delta =  target - critic_value

        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1

        self.actor.fit([state, delta], actions, verbose=0)

        self.critic.fit(state, target, verbose=0)

    def choose_action(self, observation, ball):

        ball.is_kicked = False
        state = observation[np.newaxis, :]
        probabilities = self.policy.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)
        if action <3:
            self.rotate(direction=action)
            # self.reward=-(self.position.distance_between_points(ball.position))/WIDTH/10
            self.reward=0
        elif action <5:
            self.last_position = self.position
            self.move(speed=action)
            if self.position.distance_between_points(ball.position)< self.last_position.distance_between_points(ball.position):
                self.reward = 0
            else:
                # self.reward = -(self.position.distance_between_points(ball.position))/WIDTH/10
                self.reward=0
            self.move(speed=action)

        else:
            if ball.is_kicked == True:
                self.reward = 1
            else:
                # self.reward = -(self.position.distance_between_points(ball.position))/WIDTH/10
                self.reward=0
            self.kick_ball(ball,power=action)
        return action
 

    def build_actor_critic_network(self):
        input = Input(shape=(self.input_dims,))
        delta = Input(shape=[1])
        dense1 = Dense(self.fc1_dims, activation='tanh')(input)
        dense2 = Dense(self.fc2_dims, activation='tanh')(dense1)
        probs = Dense(self.n_actions, activation='softmax')(dense2)
        values = Dense(1, activation='linear')(dense2)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out)

            return K.sum(-log_lik*delta)

        actor = Model(input=[input, delta], output=[probs])

        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)

        critic = Model(input=[input], output=[values])

        critic.compile(optimizer=Adam(lr=self.beta), loss='mean_squared_error')

        policy = Model(input=[input], output=[probs])

        return actor, critic, policy

    def save_model(self):
        self.actor.save(self.model_name)

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











