# RL DDQN agent for soccer
I have created this environment with the purpose of exploring different RL algorithms, but in the end made it to work only with DDQN on a semi-decent level while tweaking environment rewards a little bit.
We have two players and a ball in the environment, players cannot go out of bounds. If the ball gets out of bounds, episode terminates with reward 0 for both players. If goal is scored, episode terminates with reward 1 and -1, depending on which goal. 
The idea was to see how hard it is for a player to learn to defend. 
Some sample gifs are posted bellow:



![Alt Text](scoring_gif.gif)

![Alt Text](scoring2.gif)


![Alt Text](gif3.gif)
