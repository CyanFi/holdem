
from random import randint

from gym import error

from treys import Card


class Player(object):

    CHECK = 0
    CALL = 1
    RAISE = 2
    FOLD = 3

    def __init__(self, player_id, stack=1000, emptyplayer=False, playername="", reloadCount=0, roundRaiseLimit=4):
        self.player_id = player_id
        self.playername = playername

        self.reloadCount = reloadCount
        self.hand = []
        self.stack = stack
        self._init_stack = stack
        self.currentbet = 0
        self.lastsidepot = 0
        self._seat = -1
        self.handrank = -1

        # flags for table management
        self.emptyplayer = emptyplayer
        self.betting = 0
        self.isallin = False
        self.playing_hand = False
        self.playedthisround = False
        self.sitting_out = True
        self._roundRaiseLimit = roundRaiseLimit
        self._roundRaiseCount = 0

    def get_name(self):
        return self.playername

    def get_seat(self):
        return self._seat

    def set_seat(self, value):
        self._seat = value

    def reset_hand(self):
        self._hand = []
        self.playedthisround = False
        self.betting = 0
        self.isallin = False
        self.currentbet = 0
        self.lastsidepot = 0
        self.playing_hand = (self.stack != 0)

    def bet(self, bet_size):
        self.playedthisround = True
        if not bet_size:
            return
        self.stack -= (bet_size - self.currentbet)
        self.betting += (bet_size - self.currentbet)
        self.currentbet = bet_size
        if self.stack == 0:
            self.isallin = True

    def refund(self, ammount):
        self.stack += ammount

    def player_state(self):
        return (self.get_seat(), self.stack, self.playing_hand, self.betting, self.player_id)

    def reset_stack(self):
        self.stack = self._init_stack

    def update_localstate(self, table_state):
        self.stack = table_state.get('stack')
        self.hand = table_state.get('pocket_cards')

    def validate_action(self, table_state, action):
        """Check player's action. Return the move tuple"""
        self.update_localstate(table_state)
        tocall = min(table_state.get('tocall'), self.stack)
        minraise = table_state.get('minraise')
        maxraise = table_state.get('maxraise')

        [action_idx, raise_amount] = action
        raise_amount = int(raise_amount)
        action_idx = int(action_idx)

        if tocall == 0:
            assert action_idx in [Player.CHECK, Player.RAISE]
            if action_idx == Player.RAISE:
                if self._roundRaiseCount > self._roundRaiseLimit:
                    raise error.Error('raise times ({}) in this round had exceed limitation ({})'.format(self._roundRaiseCount, self._roundRaiseLimit))
                if raise_amount < minraise:
                    raise error.Error('Raise amount {} must be no less than minraise {}.'.format(raise_amount,minraise))
                if raise_amount > maxraise:
                    raise error.Error('Raise amount {} must be no greater than maxraise {}.'.format(raise_amount,maxraise))
                move_tuple = ('raise', raise_amount)
                self._roundRaiseCount += 1
            elif action_idx == Player.CHECK:
                move_tuple = ('check', 0)
            else:
                raise error.Error('invalid action ({}) must be check (0) or raise (2)'.format(action_idx))
        else:
            # to_call!=0
            if action_idx not in [Player.RAISE, Player.CALL, Player.FOLD]:
                raise error.Error('invalid action ({}) must be call (1), raise (2),  or fold (3)'.format(action_idx))
            if action_idx == Player.RAISE:
                if raise_amount < minraise:
                    raise error.Error('raise must be no less than minraise {} but {}'.format(minraise, raise_amount))
                if raise_amount > maxraise:
                    raise error.Error('raise must be no greater than maxraise {}'.format(maxraise))
                move_tuple = ('raise', raise_amount)
            elif action_idx == Player.CALL:
                move_tuple = ('call', tocall)
            elif action_idx == Player.FOLD:
                move_tuple = ('fold', -1)
            else:
                raise error.Error('Invalid action ({})!'.format(action_idx))
        return move_tuple
