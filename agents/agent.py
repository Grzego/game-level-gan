import abc


class Agent(object):

    @abc.abstractmethod
    def act(self, state):
        """
        Provided a state agent needs to return action
        it wants to take.
        """
        pass

    @abc.abstractmethod
    def observe(self, reward):
        """
        Agent can observe a reward it got
        from an action taken.
        """
        pass

    @abc.abstractmethod
    def learn(self):
        """
        Called after game episode allows agent
        to learn from gathered data.
        """
        pass

    @abc.abstractmethod
    def generator_data(self):
        """
        Returns data that generator can use
        for backward pass.
        """
        pass
