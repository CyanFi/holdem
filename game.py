import gym
import holdem
import agent
from termcolor import colored


def lets_play(env, n_seats, model_list):
    while True:
        cur_state = env.reset()
        # env.render(mode='human')
        cycle_terminal = False
        if env.episode_end:
            break

        while not cycle_terminal:
 
            actions = holdem.model_list_action(cur_state, n_seats=n_seats, model_list=model_list)
            cur_state, rews, cycle_terminal, info = env.step(actions)
            env.render(mode="human",cur_episode=i)

    print(colored("Episode ends.\n", 'magenta'))


env = gym.make('TexasHoldem-v0')
model_list = list()
increment_blind=False

# start with 2 players
env.add_player(0, stack=1000)
model_list.append(agent.idiotModel())

env.add_player(1, stack=1000)
model_list.append(agent.idiotModel())

max_episode = 1
for i in range(0, max_episode):
    lets_play(env, env.n_seats, model_list)
