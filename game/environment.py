import abc


class MultiEnvironment(object):

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

        Returns nothing
        """
        pass

    @abc.abstractmethod
    def actions(self):
        """
        Returns number of actions
        """
        pass
