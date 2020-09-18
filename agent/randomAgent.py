from collections import namedtuple
from enum import Enum
from holdem import PLAYER_STATE, COMMUNITY_STATE, STATE, ACTION, action_table
import random


class randomModel():
    def __init__(self):
        self._nothing = "test"
        self.reload_left = 2
        self.model = {"seed": 831}

    def batchTrainModel(self):
        return

    def onlineTrainModel(self):
        return

    def saveModel(self, path):
        return

    def loadModel(self, path):
        return

    def takeAction(self, state, playerid, valid_actions):
        """ random action"""
        tmp = []
        if valid_actions['fold']:
            tmp.append('fold')
        if valid_actions['check']:
            tmp.append('check')
        if valid_actions['call']:
            tmp.append('call')
        if valid_actions['raise']:
            tmp.append('raise')
        choice = random.choice(tmp)

        if choice == 'fold':
            return ACTION(action_table.FOLD, 0)
        elif choice == 'check':
            return ACTION(action_table.CHECK, 0)
        elif choice == 'call':
            return ACTION(action_table.CALL, valid_actions['call_amount'])
        elif choice == 'raise':
            if valid_actions['raise_range'][1]<valid_actions['raise_range'][0]:
                IndexError
            raise_amount = random.randint(valid_actions['raise_range'][0], valid_actions['raise_range'][1])
            return ACTION(action_table.RAISE, raise_amount)
        else:
            assert 1 == 0

    def getReload(self, state):
        """return `True` if reload is needed under state, otherwise `False`"""
        if self.reload_left > 0:
            self.reload_left -= 1
            return True
        else:
            return False
