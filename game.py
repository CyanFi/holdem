import gym
import holdem
import agent
import argparse
from termcolor import colored


def colored_output(string, color):
    if arg_list.log:
        return string
    else:
        return colored(string, color)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_episode',
                        type=int,
                        default=1,
                        help='define the number of training episodes')
    parser.add_argument('--no_blind_increment',
                        action='store_true',
                        help='define whether amount of blinds will increase in a episode')
    parser.add_argument('--log',
                        action='store_true',
                        help='define whether to log or not')
    return parser.parse_args()


def episode(env, n_seats, model_list):
    if arg_list.log:
        o_mode = 'machine'
    else:
        o_mode = 'human'
    while True:
        cur_state, cycle_terminal = env.reset()
        if cycle_terminal:
            # a cycle may terminate here because players may be "forced" to all in if they have a low stack
            env.render(mode=o_mode, cur_episode=i)
        if env.episode_end:
            break
        valid_actions = env.get_valid_actions(env._current_player)
        while not cycle_terminal:
            # in
            actions = holdem.model_list_action(cur_state, n_seats=n_seats, model_list=model_list,
                                               valid_actions=valid_actions)
            cur_state, rews, cycle_terminal, valid_actions = env.step(actions)
            env.render(mode=o_mode, cur_episode=i)

    print(colored("Episode ends.\n", 'magenta'))


if __name__ == "__main__":
    arg_list = parse()

    env = gym.make('TexasHoldem-v0')
    if arg_list.no_blind_increment:
        env.blind_increment = False
        print((colored_output("No blind increment in episodes!"), 'magenta'))
    model_list = list()
    increment_blind = False

    # start with 2 players
    env.add_player(0, stack=1000)
    model_list.append(agent.idiotModel())

    env.add_player(1, stack=1000)
    model_list.append(agent.idiotModel())

    if arg_list.log:
        env.log = True
        import random
        import sys

        log_name = 'log/' + 'game_log_' + ''.join(random.sample('123456789abcdedg', 10)) + '.log'
        print('Log file will be saved at ', log_name)
        sys.stdout = open(log_name, 'w+')
    else:
        print(colored_output("The game log is not saved", 'magenta'))

    max_episode = arg_list.max_episode
    print(colored_output("Episode: {}. Start now!".format(max_episode), 'magenta'))
    for i in range(0, max_episode):
        episode(env, env.n_seats, model_list)
