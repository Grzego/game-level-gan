

class SimplePacmanGenerator(object):
    """
    Simple generator for Pacman game.
    """

    def __init__(self, board_size, num_players):
        """
        `num_players` in range [2, 4] -- each players is put in corner.
        """
        self.board_size = board_size
        self.num_players = num_players

    def generate(self):
        """
        From random vector generate map of `board_size`.
        """
        pass

    def backward(self, agents):
        """
        Use data stored in agents for gradients calculations.
        """
        pass

