import abc


class MultiEnvironment(object):

    @abc.abstractmethod
    def players_layer_shape(self):
        """
        Returns shape of players layer used in game
        """
        pass

    @abc.abstractmethod
    def state_shape(self):
        """
        Returns shape of game input state
        """
        pass

    @abc.abstractmethod
    def reset(self, *args, **kwargs):
        """
        This function resets environment

        Returns initial state
        """
        pass

    @abc.abstractmethod
    def step(self, actions):
        """
        This function changes state of the game executing
        given actions in order

        Returns stetes and rewards for each player
        """
        pass

    @abc.abstractmethod
    def actions(self):
        """
        Returns number of actions
        """
        pass
