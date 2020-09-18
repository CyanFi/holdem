from gym import Env, error, spaces, utils
from gym.utils import seeding
import sys

from treys.evaluator import Evaluator
from treys.card import Card
from treys.deck import Deck
from termcolor import colored
from .player import Player
from .utils import hand_to_str, format_action, PLAYER_STATE, COMMUNITY_STATE, STATE


class TexasHoldemEnv(Env, utils.EzPickle):
    BLIND_INCREMENTS = [[10, 20], [20, 40], [40, 80], [80, 160], [160, 320]]

    def __init__(self, n_seats, max_limit=20000, debug=False):
        self.log = False
        n_suits = 4  # s,h,d,c
        n_ranks = 13  # 2,3,4,5,6,7,8,9,T,J,Q,K,A
        n_community_cards = 5  # flop, turn, river
        n_pocket_cards = 2
        n_stud = 5

        self.n_seats = n_seats
        # fill seats with dummy players
        self._seats = [Player(i, stack=0, emptyplayer=True) for i in range(self.n_seats)]
        self.emptyseats = self.n_seats

        self.episode_reset()

        self.observation_space = spaces.Tuple([
            spaces.Tuple([  # # **players info**
                             spaces.MultiDiscrete([
                                 1,  # (boolean) is_emptyplayer
                                 n_seats - 1,  # (numbers) number of seat
                                 max_limit,  # (numbers) stack
                                 1,  # (boolean) is_playing_hand
                                 max_limit,
                                 # (numbers) handrank, need_error_msg?  (0 could be no rank, no error_msg needed
                                 1,  # (boolean) is_playedthisround
                                 max_limit,  # (numbers) is_betting
                                 1,  # (boolean) isallin
                                 max_limit,  # (numbers) last side pot
                             ]),
                             spaces.Tuple([
                                              spaces.MultiDiscrete([  # # **players hand**
                                                  1,  # (boolean) is_avalible
                                                  n_suits,  # (catagory) suit,
                                                  n_ranks,  # (catagory) rank.
                                              ])
                                          ] * n_pocket_cards)
                         ] * n_seats),
            spaces.Tuple([
                             spaces.Discrete(n_seats - 1),  # big blind location
                             spaces.Discrete(max_limit),  # small blind
                             spaces.Discrete(max_limit),  # big blind
                             spaces.Discrete(max_limit * n_seats),  # pot amount
                             spaces.Discrete(max_limit),  # last raise
                             spaces.Discrete(max_limit),  # minimum amount to raise
                             spaces.Discrete(max_limit),  # how much needed to call by current player.
                             spaces.Discrete(n_seats - 1),  # current player seat location.
                             spaces.MultiDiscrete([  # community cards
                                 1,  # (boolean) is_avalible
                                 n_suits,  # (catagory) suit
                                 n_ranks,  # (catagory) rank
                                 1,  # (boolean) is_flopped
                             ]),
                         ] * n_stud),
        ])

        self.action_space = spaces.Tuple([
                                             spaces.MultiDiscrete([
                                                 3,  # action_id
                                                 max_limit,  # raise_amount
                                             ]),
                                         ] * n_seats)

    def add_player(self, seat_id, stack=1000):
        """Add a player to the environment seat with the given stack (chipcount)"""
        player_id = seat_id
        if player_id not in self._player_dict:
            new_player = Player(player_id, stack=stack, emptyplayer=False)
            if self._seats[player_id].emptyplayer:
                self._seats[player_id] = new_player
                new_player.set_seat(player_id)
            else:
                raise error.Error('Seat already taken.')
            self._player_dict[player_id] = new_player
            self.emptyseats -= 1

    def remove_player(self, seat_id):
        """Remove a player from the environment seat."""
        player_id = seat_id
        try:
            idx = self._seats.index(self._player_dict[player_id])
            self._seats[idx] = Player(0, stack=0, emptyplayer=True)
            del self._player_dict[player_id]
            self.emptyseats += 1
        except ValueError:
            pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def episode_reset(self):
        """reset a episode"""
        self._cycle = 0
        self._blind_index = 0
        [self._smallblind, self._bigblind] = TexasHoldemEnv.BLIND_INCREMENTS[0]
        self.blind_increment = True
        self._deck = Deck()
        self._evaluator = Evaluator()

        self.community = []
        self._round = 0
        self._button = 0
        self._discard = []

        self._side_pots = [0] * self.n_seats
        self._current_sidepot = 0  # index of _side_pots
        self._totalpot = 0
        self._tocall = 0
        self._lastraise = 0
        self._roundpot = 0

        self._player_dict = {}
        self._current_player = None
        self._last_player = None
        self._last_actions = None
        self._debug = False

        self.episode_end = False

    def reset(self):
        """
        Reset a cycle
        """
        self._reset_game()
        self._ready_players()
        self._cycle += 1
        if self.blind_increment and self._cycle % self.n_seats == 0:
            # increase blind
            self._increment_blinds()

        [self._smallblind, self._bigblind] = TexasHoldemEnv.BLIND_INCREMENTS[self._blind_index]

        alive_player = sum([1 if p.stack > 0 else 0 for p in self._seats])
        if (self.emptyseats < len(self._seats) - 1) and (alive_player > len(self._seats) // 2):
            # episode continues
            players = [p for p in self._seats if p.playing_hand]
            self._new_round_env()
            self._round = 0
            self.round_terminate = False
            self._current_player = sb = self._first_to_act(players)
            self._post_smallblind(self._current_player)
            self._current_player = bb = self._get_next_player(players, self._current_player)
            self._post_bigblind(self._current_player)
            self._current_player = self._get_next_player(players, self._current_player)
            self._tocall = self._bigblind
            self._round = 0
            self.deal_card()
            self._folded_players = []
            print(self.colored_output(
                'New Cycle starts. SB: player {}, BB: player {}. BB amount:{}'.format(sb.player_id, bb.player_id,
                                                                                      self._bigblind), 'magenta'))
            all_in_players = [p for p in self._seats if p.isallin]
            if all_in_players:
                # Player are "forced" to all in
                self._resolve_sidepots_each_round(self._seats)
                self._two_players_all_in()
                self.cycle_checkout(self._seats)
                if [p for p in self._seats if p.stack == 0]:
                    # A player is forced to all-in ,and he loses in all-in bet.
                    self.episode_end = True
                return self._get_current_reset_returns(), True

        else:
            self.episode_end = True
            return self._get_current_reset_returns(), True
        return self._get_current_reset_returns(), False

    def step(self, actions):
        """ Player make actions in a round (t -> t+1)
            @param: actions the list of actions that player selected.
            ACTIONS_ENUM {
              CHECK = 0,
              CALL = 1,
              RAISE = 2,
              FOLD = 3
            }
            RAISE_AMT = [minraise, maxraise]
           """
        cycle_terminate = False
        self.round_terminate = False
        if len(actions) != len(self._seats):
            raise error.Error('actions must be same shape as number of seats.')

        if self._current_player is None:
            raise error.Error('Round cannot be played without 2 or more players.')

        if self._round == 4:
            raise error.Error('Rounds already finished, needs to be reset.')

        # log last player and his actions
        self._last_player = self._current_player
        self._last_actions = actions

        alive_players = [p for p in self._seats if p.playing_hand]
        move = self._current_player.validate_action(self._output_state(self._current_player),
                                                    actions[self._current_player.player_id])

        if move[0] == 'call':
            self._player_bet(self._current_player, self._tocall - self._current_player.currentbet)
            next_player = self._get_next_player(alive_players, self._current_player)
            if self._current_player.isallin or next_player.isallin:
                # if player calls all-in or after call, the player himself is all-in, deal the cards directly
                self._resolve_sidepots_each_round(self._seats)
                self._two_players_all_in()
                self.cycle_checkout(self._seats)
                if [p for p in self._seats if p.stack == 0]:
                    # A player is forced to all-in ,and he loses in all-in bet.
                    self.episode_end = True
                return self._get_current_step_returns(True)
            self._current_player = next_player

        elif move[0] == 'check':
            self._player_bet(self._current_player, 0)
            self._current_player = self._get_next_player(alive_players, self._current_player)

        elif move[0] == 'raise':
            self._player_bet(self._current_player, move[1])
            for p in alive_players:
                # set other players to "unplayed" in this round
                if p != self._current_player:
                    p.playedthisround = False
            self._current_player = self._get_next_player(alive_players, self._current_player)

        elif move[0] == 'fold':
            self._current_player.playing_hand = False
            folded_player = self._current_player
            self._current_player = self._get_next_player(alive_players, self._current_player)
            alive_players.remove(folded_player)
            self._folded_players.append(folded_player)
            # For two players, it is the end of cycle
            self._resolve_sidepots_each_round(alive_players + self._folded_players)
            self.cycle_checkout(alive_players)
            cycle_terminate = True

        if move[0] != 'fold' and self._current_player.playedthisround and len(
                [p for p in alive_players if not p.isallin]) >= 1:
            # end of round
            self.round_checkout(alive_players + self._folded_players)

        if self._round == 4:
            cycle_terminate = True
            self.cycle_checkout(alive_players)

        if cycle_terminate:
            valid_actions = []
        else:
            valid_actions = self.get_valid_actions(self._current_player)

        return self._get_current_step_returns(cycle_terminate, valid_actions)

    def get_valid_actions(self, cur_player):
        table_state = self._output_state(cur_player)
        tocall = min(table_state.get('tocall'), cur_player.stack)
        minraise = table_state.get('minraise')
        maxraise = table_state.get('maxraise')

        if tocall == 0:
            if self._last_player and self._last_player.isallin:
                #  opponent is all-in
                return {'fold': False, 'check': False, 'call': True, 'raise': False, 'call_amount': tocall}
            if maxraise - minraise >= 0:
                return {'fold': False, 'check': True, 'call': False, 'raise': True, 'raise_range': [minraise, maxraise]}
            else:
                return {'fold': False, 'check': True, 'call': False, 'raise': False}
        else:
            # to_call>0
            if self._last_player and self._last_player.isallin:
                #  opponent is all-in
                return {'fold': True, 'check': False, 'call': True, 'raise': False, 'call_amount': tocall}
            elif maxraise - minraise >= 0:
                return {'fold': True, 'check': False, 'call': True, 'raise': True, 'call_amount': tocall,
                        'raise_range': [minraise, maxraise]}
            else:
                return {'fold': True, 'check': False, 'call': True, 'raise': False, 'call_amount': tocall}

    def print_round_info(self, cur_episode=-1000):
        if self.round_terminate:
            round = self._round
        else:
            round = self._round + 1
        print('In episode {}, cycle {}, round {}, total pot: {}.'
              .format(self.colored_output(cur_episode + 1, 'red'), self.colored_output(self._cycle, 'green'),
                      self.colored_output(round, 'blue'),
                      self._totalpot))

    def render(self, mode='machine', close=False, cur_episode=-1000):

        self.print_round_info(cur_episode)
        if self._last_actions is not None:
            pid = self._last_player.player_id
            print('Player {}\'s action:'.format(pid) + format_action(self._last_player, self._last_actions[pid], mode))

        state = self._get_current_state()

        print('Community card:')
        print('-' + hand_to_str(state.community_card, mode))
        print('Players status:')
        for idx, playerstate in enumerate(state.player_states):
            print(
                'Player #{}---{}stack: {} all-in: {}.'.format(self.colored_output(idx, 'cyan'),
                                                              hand_to_str(playerstate.hand, mode),
                                                              self._seats[idx].stack, self._seats[idx].isallin))
        print("")

    def _two_players_all_in(self):
        self._resolve_sidepots_each_round(self._seats)
        while len(self.community) < 5:
            self._discard.append(self._deck.draw(1))
            self.community.append(self._deck.draw(1))

    def round_checkout(self, players):
        """
        End the round and start a new round
        """
        # resolve sidepot
        self._resolve_sidepots_each_round(players + self._folded_players)

        # Start a new round
        self._current_player = self._first_to_act(players)
        self._new_round_env()
        self.deal_card()
        self.round_terminate = True

    def deal_card(self):
        """
        Deal card to start the next round
        """
        if self._round == 0:
            self._deal()
        elif self._round == 1:
            self._flop()
        elif self._round == 2:
            self._turn()
        elif self._round == 3:
            self._river()

    def _increment_blinds(self, indicator=True):
        if indicator:
            self._blind_index = min(self._blind_index + 1, len(TexasHoldemEnv.BLIND_INCREMENTS) - 1)
            [self._smallblind, self._bigblind] = TexasHoldemEnv.BLIND_INCREMENTS[self._blind_index]

    def _post_smallblind(self, player):
        """ choose player to act as the small blind """
        self._player_bet(player, self._smallblind)
        player.playedthisround = False

    def _post_bigblind(self, player):
        """ choose player to act as the big blind """
        self._player_bet(player, self._bigblind)
        player.playedthisround = False
        self._lastraise = self._bigblind

    def _player_bet(self, player, player_bet):
        total_bet = min(player.stack, player_bet) + player.currentbet

        self._roundpot = max(self._roundpot, total_bet)
        self._totalpot += (total_bet - player.currentbet)

        player.bet(total_bet)  # update acumulative bet (not add another bet)

        self._tocall = max(self._tocall, total_bet)
        # if self._tocall > 0:
        # self._tocall = max(self._tocall, self._bigblind)
        self._lastraise = max(self._lastraise, total_bet - self._lastraise)

    def _first_to_act(self, players):
        if self._round == 0 and len(players) == 2:
            return self._get_next_player(sorted(
                players + [self._seats[self._button]], key=lambda x: x.get_seat()),
                self._seats[self._button])
        try:
            first = [player for player in players if player.get_seat() > self._button][0]
        except IndexError:
            first = players[0]
        return first

    def _get_next_player(self, players, current_player):
        """ get next player """
        idx = players.index(current_player)
        return players[(idx + 1) % len(players)]

    def _deal(self):
        for player in self._seats:
            if player.playing_hand:
                player.hand = self._deck.draw(2)

    def _flop(self):
        self._discard.append(self._deck.draw(1))
        self.community = self._deck.draw(3)

    def _turn(self):
        self._discard.append(self._deck.draw(1))
        self.community.append(self._deck.draw(1))

    def _river(self):
        self._discard.append(self._deck.draw(1))
        self.community.append(self._deck.draw(1))

    def _ready_players(self):
        for p in self._seats:
            p.isallin = False
            if not p.emptyplayer and p.sitting_out:
                p.sitting_out = False
                p.playing_hand = True

    def _resolve_sidepots_each_round(self, players_playing):
        players = [p for p in players_playing if p.currentbet]
        if self._debug:
            print('[DEBUG] current bets: ', [p.currentbet for p in players])
            print('[DEBUG] playing hand: ', [p.playing_hand for p in players])
        if not players:
            return
        try:
            smallest_bet = min([p.currentbet for p in players if p.playing_hand])
        except ValueError:
            for p in players:
                self._side_pots[self._current_sidepot] += p.currentbet
                p.currentbet = 0
            return

        smallest_players_allin = [p for p, bet in zip(players, [p.currentbet for p in players]) if
                                  bet == smallest_bet and p.isallin]

        for p in players:
            self._side_pots[self._current_sidepot] += min(smallest_bet, p.currentbet)
            p.currentbet -= min(smallest_bet, p.currentbet)
            p.lastsidepot = self._current_sidepot

        if smallest_players_allin:
            self._current_sidepot += 1
            self._resolve_sidepots_each_round(players)
        if self._debug:
            print('[DEBUG] sidepots: ', self._side_pots)

    def _new_round_env(self):
        for player in self._player_dict.values():
            player.currentbet = 0
            player._roundRaiseCount = 0
            player.playedthisround = False
        self._round += 1
        self._tocall = 0
        self._lastraise = 0
        self._roundpot = 0

        # print(colored("New round starts.", 'magenta'))

    def cycle_checkout(self, players):
        if len(players) == 1:
            # winning player get the refund
            players[0].refund(sum(self._side_pots))
            self._totalpot = 0
        else:
            # compute hand ranks
            for player in players:
                player.handrank = self._evaluator.evaluate(player.hand, self.community)

            # trim side_pots to only include the non-empty side pots
            temp_pots = [pot for pot in self._side_pots if pot > 0]

            # compute who wins each side pot and pay winners
            for pot_idx, _ in enumerate(temp_pots):
                # find players involved in given side pot, compute the winner(s)
                pot_contributors = [p for p in players if p.lastsidepot >= pot_idx]
                winning_rank = min([p.handrank for p in pot_contributors])
                winning_players = [p for p in pot_contributors if p.handrank == winning_rank]
                print(self.colored_output("Cycle winner: {}".format(winning_players[0].player_id), 'magenta'))
                for player in winning_players:
                    split_amount = int(self._side_pots[pot_idx] / len(winning_players))
                    if self._debug:
                        print('[DEBUG] Player', player.player_id, 'wins side pot (',
                              int(self._side_pots[pot_idx] / len(winning_players)), ')')
                    player.refund(split_amount)
                    self._side_pots[pot_idx] -= split_amount

                # any remaining chips after splitting go to the winner in the earliest position
                if self._side_pots[pot_idx]:
                    earliest = self._first_to_act([player for player in winning_players])
                    earliest.refund(self._side_pots[pot_idx])

    def _reset_game(self):
        playing = 0
        for player in self._seats:
            if not player.emptyplayer and not player.sitting_out:
                player.reset_hand()
                playing += 1
        self.community = []
        self._current_sidepot = 0
        self._totalpot = 0
        self._side_pots = [0] * len(self._seats)
        self._deck.shuffle()

        if playing:
            self._button = (self._button + 1) % len(self._seats)
            while not self._seats[self._button].playing_hand:
                self._button = (self._button + 1) % len(self._seats)

    def _output_state(self, current_player):
        next_player = self._get_next_player(self._seats, self._current_player)
        return {
            'players': [player.player_state() for player in self._seats],
            'community': self.community,
            'my_seat': current_player.get_seat(),
            'pocket_cards': current_player.hand,
            'pot': self._totalpot,
            'button': self._button,
            'tocall': (self._tocall - current_player.currentbet),
            'stack': current_player.stack,
            'bigblind': self._bigblind,
            'player_id': current_player.player_id,
            'lastraise': self._lastraise,
            'minraise': max(self._bigblind, self._lastraise + self._tocall),
            "maxraise": min(current_player.stack, next_player.stack)

        }

    def _pad(self, l, n, v):
        if (not l) or (l is None):
            l = []
        return l + [v] * (n - len(l))

    def _get_current_state(self):
        player_states = []
        for player in self._seats:
            player_features = PLAYER_STATE(
                int(player.emptyplayer),
                int(player.get_seat()),
                int(player.stack),
                int(player.playing_hand),
                int(player.handrank),
                int(player.playedthisround),
                int(player.betting),
                int(player.isallin),
                int(player.lastsidepot),
                0,
                self._pad(player.hand, 2, -1)
            )
            player_states.append(player_features)

        community_states = COMMUNITY_STATE(
            int(self._button),
            int(self._smallblind),
            int(self._bigblind),
            int(self._totalpot),
            int(self._lastraise),
            int(self._roundpot),
            int(self._tocall - self._current_player.currentbet),
            int(self._current_player.player_id)
        )
        return STATE(tuple(player_states), community_states, self._pad(self.community, 5, -1))

    def _get_current_reset_returns(self):
        return self._get_current_state()

    def _get_current_step_returns(self, terminal, valid_actions=[]):
        obs = self._get_current_state()
        rew = [player.stack for player in self._seats]
        return obs, rew, terminal, valid_actions

    def colored_output(self, string, color):
        if self.log:
            return string
        else:
            return colored(string, color)
