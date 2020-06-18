import pygame
import math
import random
from world import World, Pitch
from player import Player, Goalkeeper
from ball import Ball
from team import Team
import numpy as np
from helpers import draw_environment, choose_action
from matplotlib import pyplot as plt
from point import Point

WIDTH = 800
HEIGHT = 600
PLAYER_SIZE = 35
BALL_SIZE = 15
NUM_EPISODES=50000
EPISODE_LENGTH=10000

ball = Ball()
pitch = Pitch()
team1 = Team(pitch)
team2 = Team(pitch, name='Team2')
players_team_1 = team1.lineup
players_team_2 = team2.lineup
pl_team1 = list(players_team_1)
pl_team2 = list(players_team_2)
game_display = pygame.display.set_mode((WIDTH,HEIGHT))
clock = pygame.time.Clock()


def main():
        pitch = Pitch()
        score_history = []

        for i in range(NUM_EPISODES):
            print(f"episode {i} started")
            done=False
            k=0
            cumulative_reward_team1=[]
            cumulative_reward_team2=[]
            [player.reset_position() for player in players_team_1]
            [player.reset_position() for player in players_team_2]

            ball.reset_position()
            goal1=False
            goal2=False
            for k in range(EPISODE_LENGTH):
                terminal_ind=False
                team1.reset_reward()
                team2.reset_reward()
                if i%30==0:
                    draw_environment(pitch, ball, *players_team_1, *players_team_2)

                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        print(pygame.mouse.get_pos()) 
                        ball.position.x = pygame.mouse.get_pos()[0]
                        ball.position.y = pygame.mouse.get_pos()[1]
                    elif event.type == pygame.MOUSEBUTTONUP:
                        Player.click = False
                        
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()

                clock.tick(500)
                if pitch.is_goal(ball)==1:
                     goal1 = True 
                elif pitch.is_goal(ball)==2:
                     goal2 = True
                if k == EPISODE_LENGTH-1 or pitch.is_goal(ball)==1 or pitch.is_goal(ball)==2 or pitch.out_of_bounds(ball)==True:
                    print("terminal")
                    terminal_ind=True
                    done=True
                else:
                    done=False

                action_reward_team_1 = [choose_action(player,pl_team2[0], ball, goal1, goal2) for player in pl_team1]
                action_reward_team_2 = [choose_action(player,pl_team1[0], ball, goal1, goal2) for player in pl_team2]

                observation_ =players_team_1[0].return_observation(players_team_2[0], ball)
                observation_team_2_ = players_team_2[0].return_observation(players_team_1[0], ball)
                pl_team1[0].remember(action_reward_team_1[0][0], action_reward_team_1[0][1], pl_team1[0].reward, observation_, int(done))
                pl_team2[0].remember(action_reward_team_2[0][0], action_reward_team_2[0][1], pl_team2[0].reward, observation_team_2_, int(done))


                # pl_team1[0].learn(action_reward[0][0], action_reward[0][1], action_reward[0][2], observation_, int(done))
                pl_team1[0].learn()
                pl_team2[0].learn()
                k+=1
                cumulative_reward_team1.append(pl_team1[0].reward)
                cumulative_reward_team2.append(pl_team2[0].reward)

                pl_team1[0].reward=0
                pl_team2[0].reward=0

                pl_team1[0].last_observation = pl_team1[0].position
                pl_team2[0].last_observation = pl_team2[0].position
                if terminal_ind==True:
                    break


            if i % 50==0 and i>0:
                players_team_1[0].save_model()
                players_team_1[0].alpha = players_team_1[0].alpha * 0.98
                players_team_1[0].beta = players_team_1[0].beta * 0.98
                print("saved model")

                

            cumulative_reward1=sum(np.array(cumulative_reward_team1))
            cumulative_reward2=sum(np.array(cumulative_reward_team2))
            epsilon = players_team_1[0].epsilon
            print(f"Reward for episode {i} for team1 is {cumulative_reward1}, for team2 is {cumulative_reward2},  epsilon is {epsilon}")




if __name__ == '__main__':
    main()

