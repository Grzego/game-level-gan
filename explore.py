import os
import random
import pyforms
from pyforms.controls import *
import torch
from torch import optim
import torch.multiprocessing as mp
import numpy as np
from tensorboardX import SummaryWriter

from game import Race, RaceCar
from agents import PPOAgent
from generators import RaceTrackGenerator
from networks import LSTMPolicy, RaceWinnerDiscriminator
from utils import find_next_run_dir, find_latest, one_hot, device


resume = os.path.join('learned')
num_players = 2
num_segments = 128
latent = 16


class GUI(pyforms.BaseWidget):
    def __init__(self):
        super().__init__('Explore generator')

        for i in range(latent):
            self.__dict__[f'_slider_{i}'] = ControlSlider(f'{i}', minimum=-5000, maximum=5000)
            self.__dict__[f'_slider_{i}'].changed_event = self._regenerate

        self._sample = ControlButton('Sample')
        self._sample.value = self._sample_button

        self._zero = ControlButton('Zero')
        self._zero.value = self._zero_button

        self._board = ControlImage()

        self.formset = [{
            '0-15': [f'_slider_{i}' for i in range(0, 16)],
            # '16-31': [f'_slider_{i}' for i in range(16, 32)],
            # '32-47': [f'_slider_{i}' for i in range(32, 48)],
            # '48-63': [f'_slider_{i}' for i in range(48, 64)]
        }, ' ', '||', ['_board', '=', ['_sample', '||', '_zero']]]

        self.noise = torch.randn((1, latent)).to(device)
        self.board = None
        self._regenerate()

    def _regenerate(self):
        with torch.no_grad():
            for i in range(latent):
                self.noise[0, i] = float(self.__dict__[f'_slider_{i}'].value) / 1000.

            self.board = generator.network(self.noise, num_segments, t=100.)[0].detach_()

            game.reset(self.board)
            self._board.value = game.prettier_tracks(top_n=1)[0, :, :, :3]
            self._board.repaint()

    def _reset(self):
        for i in range(latent):
            self.__dict__[f'_slider_{i}'].changed_event = lambda: ()
            self.__dict__[f'_slider_{i}'].value = int(self.noise[0, i].item() * 1000.)
            self.__dict__[f'_slider_{i}'].changed_event = self._regenerate

        self._regenerate()

    def _zero_button(self):
        with torch.no_grad():
            self.noise = torch.zeros((1, latent)).to(device)
            self._reset()

    def _sample_button(self):
        with torch.no_grad():
            self.noise = torch.randn((1, latent)).to(device)
            self._reset()


def main():
    global generator, discriminator, agents, game

    # create agents with LSTM policy network
    cars = [RaceCar(max_speed=60., acceleration=2., angle=40.),
            RaceCar(max_speed=60., acceleration=1., angle=80.)]
    game = Race(timeout=60. * 10., framerate=1. / 20., cars=cars)

    agents = [PPOAgent(game.actions,
                       LSTMPolicy(game.state_shape()[0], game.actions),
                       lr=5e-5, discount=0.99, eps=0.1, asynchronous=True)
              for _ in range(game.num_players)]

    # load agents if resuming
    # if resume:
    #     for i, a in enumerate(agents):
    #         path = find_latest(resume, 'agent_{}_*.pt'.format(i))
    #         print(f'Resuming agent {i} from path "{path}"')
    #         a.network.load_state_dict(torch.load(path))
    #         a.old_network.load_state_dict(torch.load(path))

    # create discriminator
    # discriminator = RaceWinnerDiscriminator(num_players, lr=1e-5, asynchronous=True)

    # if resume:
    #     path = find_latest(resume, 'discriminator_*.pt')
    #     print(f'Resuming discriminator from path "{path}"')
    #     discriminator.network.load_state_dict(torch.load(path))

    # create generator
    generator = RaceTrackGenerator(latent, lr=1e-5, asynchronous=True)

    if resume:
        path = find_latest(resume, 'generator_[0-9]*.pt')
        print(f'Resuming generator from path "{path}"')
        generator.network.load_state_dict(torch.load(path, map_location=device))
        generator.deterministic = True

    # create window
    pyforms.start_app(GUI)


if __name__ == '__main__':
    main()
