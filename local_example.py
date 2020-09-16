import gym
import holdem
import agent
from termcolor import colored


def lets_play(env, n_seats, model_list):
    while True:
        cur_state = env.reset()
        env.render(mode='human')
        cycle_terminal = False
        if env.episode_end:
            break

        while not cycle_terminal:
            # play safe actions, check when no one else has raised, call when raised.
            # print(">>> Debug Information ")
            # print("state(t)")
            # for p in cur_state.player_states:
            #     print(p)
            # print(cur_state.community_state)
            env.print_round_info(i)
            actions = holdem.model_list_action(cur_state, n_seats=n_seats, model_list=model_list)
            cur_state, rews, cycle_terminal, info = env.step(actions)

            # print("action(t), (CALL=1, RAISE=2, FOLD=3 , CHECK=0, [action, amount])")
            # print(actions)

            # print("reward(t+1)")
            # print(rews)
            # print("<<< Debug Information ")
            env.render(mode="human", cur_episode=i)
        # print("final state")
        # print(cur_state)

    print(colored("Episode ends.\n", 'magenta'))


env = gym.make('TexasHoldem-v0')

model_list = list()

# start with 2 players
env.add_player(0, stack=1000)
model_list.append(agent.idiotModel())

env.add_player(1, stack=1000)
model_list.append(agent.idiotModel())

max_episode = 1
for i in range(0, max_episode):
    lets_play(env, env.n_seats, model_list)
