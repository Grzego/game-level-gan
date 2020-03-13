import pyforms
from pyforms.controls import *
import torch

from games import race_game as game
from games import RaceConfig
from generators import RaceTrackGenerator
from utils import find_latest, device

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--generator', default=None, required=True, type=str)
args = parser.parse_args()


# REQUIRES: pyforms==3.0.0
class GUI(pyforms.BaseWidget):
    def __init__(self, latent, generator):
        super().__init__('Explore generator')
        self.latent = latent
        self.generator = generator

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
            for i in range(self.latent):
                self.noise[0, i] = float(self.__dict__[f'_slider_{i}'].value) / 1000.

            self.board = self.generator.network(self.noise, RaceConfig.max_segments, t=100.)[0].detach_()

            game.reset(self.board)
            self._board.value = game.prettier_tracks(top_n=1)[0, :, :, :3]
            self._board.repaint()

    def _reset(self):
        for i in range(self.latent):
            self.__dict__[f'_slider_{i}'].changed_event = lambda: ()
            self.__dict__[f'_slider_{i}'].value = int(self.noise[0, i].item() * 1000.)
            self.__dict__[f'_slider_{i}'].changed_event = self._regenerate

        self._regenerate()

    def _zero_button(self):
        with torch.no_grad():
            self.noise = torch.zeros((1, self.latent)).to(device)
            self._reset()

    def _sample_button(self):
        with torch.no_grad():
            self.noise = torch.randn((1, self.latent)).to(device)
            self._reset()


def main():
    # create generator
    path = find_latest(args.generator, 'generator_[0-9]*.pt')
    print(f'Resuming generator from path "{path}"')
    generator = RaceTrackGenerator.from_file(path)
    latent = generator.latent_size

    class GUIEx(GUI):
        def __init__(self):
            super().__init__(latent, generator)

    # create window
    pyforms.start_app(GUIEx)


if __name__ == '__main__':
    main()
