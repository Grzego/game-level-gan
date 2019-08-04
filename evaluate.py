import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn

from game import Race, RaceCar, cars, game
from agents import PPOAgent
from generators import RaceTrackGenerator
from policies import LSTMPolicy

from utils import find_latest, one_hot, device


resume_agents = os.path.join('learned')
resume_generator = os.path.join('experiments', 'run-0')
num_players = 2
num_segments = 128
latent = 16
eval_size = 8 * 8
trials = 50


def run_evaluation(agents, game, boards, name=''):
    winners = 0.
    for t in range(trials):
        states, any_valid = game.reset(boards)
        print(f'\r[{t + 1:2d}/{trials:2d}] {name} boards eval...')
        step = 0
        while any_valid and not game.finished():
            print(f'\r[{step:4d}]', end='')
            step += 1
            actions = torch.stack([a.act(s) for a, s in zip(agents, states)], dim=0)
            states, rewards = game.step(actions)
            for a, r in zip(agents, rewards):
                a.observe(r)
        print()

        for a in agents:
            a.reset()

        winners += one_hot(game.winners() + 1, num_classes=num_players + 1).float()
    winners /= trials

    print(winners.float().mean(0))
    print(winners.float().std(0))

    plt.subplot(1, 2, 1)
    plt.hist(winners[:, 1].float().cpu().numpy(), bins=trials + 1, range=(0, 1), label='player 1')
    plt.legend()
    plt.xlim(0., 1.)
    plt.subplot(1, 2, 2)
    plt.hist(winners[:, 2].float().cpu().numpy(), bins=trials + 1, range=(0, 1), label='player 2')
    plt.xlim(0., 1.)
    # plt.show()
    plt.legend()
    plt.savefig(f'{name.lower()}.png')
    plt.close()


@torch.no_grad()
def main():
    seaborn.set()

    # create agents with LSTM policy network
    agents = [PPOAgent(game.actions,
                       LSTMPolicy(game.state_shape()[0], game.actions),
                       lr=5e-5, discount=0.99, eps=0.1, asynchronous=True)
              for _ in range(game.num_players)]

    # load agents if resuming
    if resume_agents:
        for i, a in enumerate(agents):
            path = find_latest(resume_agents, 'agent_{}_*.pt'.format(i))
            print(f'Resuming agent {i} from path "{path}"')
            a.network.load_state_dict(torch.load(path))
            a.old_network.load_state_dict(torch.load(path))
            a.network.to(device)
            a.old_network.to(device)

            torch.save(a.network.state_dict(), 'agent_{}.pt'.format(i))

    # create generator
    generator = RaceTrackGenerator(latent, lr=1e-5, asynchronous=True)

    if resume_generator:
        path = find_latest(resume_generator, 'generator_[0-9]*.pt')
        print(f'Resuming generator from path "{path}"')
        generator.network.load_state_dict(torch.load(path))
        generator.network.to(device)

    # agents on own boards
    own_boards = torch.zeros(eval_size, num_segments, 2, device=device)
    for i in range(0, num_segments, 16):
        own_boards[:, i: i + 16, 0] = 2 * ((i // 16) % 2) - 1

    run_evaluation(agents, game, own_boards, name='Own')

    # # agents on random boards
    val = 1.
    random_boards = torch.zeros(eval_size, num_segments, 2, device=device)
    random_boards[:, :, 0].uniform_(-val, val)

    run_evaluation(agents, game, random_boards, name='Random')

    # generated dummy boards
    dummy_generator = RaceTrackGenerator(latent, lr=1e-5, asynchronous=True)
    generated_boards = dummy_generator.generate(num_segments, eval_size)

    run_evaluation(agents, game, generated_boards, name='Dummy')

    # generated boards
    generated_boards = generator.generate(num_segments, eval_size, t=10.)

    run_evaluation(agents, game, generated_boards, name='Generated')


if __name__ == '__main__':
    main()
