from collections import namedtuple
from enum import Enum
from holdem import PLAYER_STATE, COMMUNITY_STATE, STATE, ACTION, action_table
import random


class idiotModel():
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

    def takeAction(self, state, playerid):
        """ random action"""
        if state.community_state.to_call > 0:
            if random.random() > 0.7:
                return ACTION(action_table.FOLD, 0)
            else:
                return ACTION(action_table.CALL, state.community_state.to_call)
        else:
            random_num = random.random()
            if random_num > 0.9:
                return ACTION(action_table.RAISE, state.player_states[playerid].stack)
            elif random_num > 0.7:
                return ACTION(action_table.RAISE, 50)
            else:
                return ACTION(action_table.CHECK, 0)

    def getReload(self, state):
        """return `True` if reload is needed under state, otherwise `False`"""
        if self.reload_left > 0:
            self.reload_left -= 1
            return True
        else:
            return False
