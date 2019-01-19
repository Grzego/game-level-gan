import os
import gc
import random
import signal
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

# learned_agents = None  # os.path.join('..', 'experiments', '013', 'run-4')
# learned_agents = os.path.join('..', 'experiments', '015', 'run-2')
learned_agents = os.path.join('learned')
learned_discriminator = None  # os.path.join('experiments', 'run-5')
# learned_discriminator = os.path.join('experiments', 'run-6')
learned_generator = None  # os.path.join('experiments', 'run-5')
# learned_generator = os.path.join('experiments', 'run-3')
resume_segments = 128
# resume_segments = 128
num_players = 2
batch_size = 32
# batch_size = 32 - 4 * 6
max_segments = 128
num_proc = 1
trials = 6
# trials = 1
latent = 16
observation_size = 18
generator_batch_size = 64
generator_train_steps = 1


def predefined_tracks():
    tracks = torch.zeros(6, max_segments, 2, device=device)

    # zig-zag
    t0 = random.randint(0, 15)
    for i in range(t0, max_segments, 16):
        tracks[0, i: i + 16, 0] = 2. * ((i // 16) % 2) - 1.

    # one sharp turn
    t1 = random.randint(0, 100)
    tracks[1, t1: t1 + 20, 0] = 1.

    # S track
    t2 = random.randint(0, 70)
    tracks[2, t2: t2 + 12, 0] = 1.
    tracks[2, t2 + 30: t2 + 42, 0] = -1.

    # large U turn
    t3 = [random.randint(0, 5) for _ in range(7)]
    offs = [0, 10, 20, 40, 50, 70, 100]
    for o, t in zip(offs, t3):
        tracks[3, t + o: t + o + 5, 0] = 1.

    # small bumpy turn
    t4 = random.randint(0, 90)
    tracks[4, t4: t4 + 6, 0] = 1.
    tracks[4, t4 + 6: t4 + 18, 0] = -1.
    tracks[4, t4 + 18: t4 + 24, 0] = 1.

    # immediate turn
    t5 = random.randint(30, 60)
    tracks[5, :20, 0] = 1.
    tracks[5, -t5: -t5 + 20, 0] = -1.

    return tracks


def train(generator: RaceTrackGenerator, discriminator: RaceWinnerDiscriminator,
          agents: [PPOAgent], result_queue: mp.Queue, pid: int, run_path: str):
    print(f'{pid:3d} -- Training started...')

    # create agents with LSTM policy network
    cars = [RaceCar(max_speed=60., acceleration=2., angle=40.),
            RaceCar(max_speed=60., acceleration=1., angle=80.)]
    game = Race(timeout=60. * 10., framerate=1. / 20., cars=cars, observation_size=observation_size)

    agents = [PPOAgent(game.actions,
                       LSTMPolicy(game.state_shape()[0], game.actions),
                       lr=5e-5, discount=0.99, eps=0.1, asynchronous=True)
              for _ in range(game.num_players)]

    del game

    # load agents if resuming
    if learned_agents:
        for i, a in enumerate(agents):
            path = find_latest(learned_agents, 'agent_{}_*.pt'.format(i))
            print(f'Resuming agent {i} from path "{path}"')
            a.network.load_state_dict(torch.load(path))
            a.old_network.load_state_dict(torch.load(path))
            a.network.cuda()
            a.old_network.cuda()

    # create discriminator
    discriminator = RaceWinnerDiscriminator(num_players, lr=1e-5, asynchronous=True)

    if learned_discriminator:
        path = find_latest(learned_discriminator, 'discriminator_[0-9]*.pt')
        print(f'Resuming discriminator from path "{path}"')
        discriminator.network.load_state_dict(torch.load(path))
        discriminator.network.cuda()

    # create generator
    generator = RaceTrackGenerator(latent, lr=1e-5, asynchronous=True)

    if learned_generator:
        path = find_latest(learned_generator, 'generator_[0-9]*.pt')
        print(f'Resuming generator from path "{path}"')
        generator.network.load_state_dict(torch.load(path))
        generator.network.cuda()

    # setup optimizers
    generator.async_optim(optim.Adam(generator.network.parameters(), lr=1e-5, betas=(0.3, 0.9)))  # dampening?
    if learned_generator:
        path = find_latest(learned_generator, 'generator_opt_[0-9]*.pt')
        print(f'Resuming generator optimizer from path "{path}"')
        generator.optimizer.load_state_dict(torch.load(path))

    discriminator.async_optim(optim.Adam(discriminator.network.parameters(), lr=2e-5, betas=(0.5, 0.9)))
    if learned_discriminator:
        path = find_latest(learned_discriminator, 'discriminator_opt_[0-9]*.pt')
        print(f'Resuming discriminator optimizer from path "{path}"')
        discriminator.optimizer.load_state_dict(torch.load(path))

    for agent in agents:
        agent.async_optim(optim.Adam(agent.network.parameters(), lr=1e-5))  # 1e-5 start, 2e-6 later

    # params
    num_segments = 2 if not learned_agents else resume_segments
    finish_mean = 0.
    episode = -1
    beta = 0.05  # 0.1 if num_segments < max_segments else 0.  # 0.1 while agents training, 0.001 when agents are ready

    # create game
    # cars = [RaceCar(max_speed=60., acceleration=4., angle=60.),
    #         RaceCar(max_speed=60., acceleration=2., angle=90.)]

    cars = [RaceCar(max_speed=60., acceleration=4., angle=40.),
            RaceCar(max_speed=60., acceleration=1., angle=80.)]
    # game = Race(timeout=3. + num_segments / 20., framerate=1. / 20., cars=cars)
    game = Race(timeout=3. + num_segments / 5., framerate=1. / 20., cars=cars, observation_size=observation_size)

    result = {}
    while True:
        episode += 1
        if episode % 30 == 0:
            gc.collect()
            print(f'{pid:3d} -- episode {episode}')

        # --training generator - generate boards
        boards = generator.generate(num_segments, batch_size).detach()
        # boards = torch.cat((boards, predefined_tracks()), dim=0)

        # -- training agents
        # boards = torch.zeros(batch_size, num_segments, 2, device=device)
        # boards[:, :, 0].uniform_(-1., 1.)

        # concat predefined tracks
        # boards = torch.cat((predefined_tracks()[:, :num_segments, :],
        #                     predefined_tracks()[:, :num_segments, :],
        #                     predefined_tracks()[:, :num_segments, :],
        #                     predefined_tracks()[:, :num_segments, :],
        #                     boards), dim=0)
        # boards = torch.cat((boards, predefined_tracks()[:, :num_segments, :]), dim=0)

        # run agents to find who wins
        boards = torch.cat((boards, -boards), dim=0)  # mirror levels to train more robust agents

        # -- training generator
        rboards = boards.repeat(trials, 1, 1)

        # -- training agents
        # rboards = boards

        states, any_valid = game.reset(rboards)
        # game.record(random.randint(0, rboards.size(0) - 1))
        # game.record(random.choice([0, 2, 5]))
        game.record(0)

        # bds = torch.zeros(batch_size, num_segments, 2).cuda()
        # bds[:, :, 0].uniform_(-1., 1.)
        # game.reset(bds)
        # game.play()
        # exit(0)

        # only when training generator/discriminator
        with torch.no_grad():
            while any_valid and not game.finished():
                actions = torch.stack([a.act(s, training=False) for a, s in zip(agents, states)], dim=0)
                # print(actions)
                states, rewards = game.step(actions)
                # for a, r in zip(agents, rewards):
                #     a.observe(r)

        # update agent policies
        # for i, a in game.iterate_valid(agents):
        #     aloss, mean_val = a.learn()
        #     result[f'agents/agent_{i}/loss'] = aloss
        #     result[f'agents/agent_{i}/mean_val'] = mean_val

        for a in agents:
            a.reset()

        cur_mean = game.finishes.float().mean().item()
        finish_mean = 0.9 * finish_mean + 0.1 * cur_mean
        result['game/finishes'] = cur_mean

        # if finish_mean >= 0.95 and num_segments < max_segments:  # was 0.9 (just to remember xD)
        #     # increase number of segments and reset mean
        #     num_segments += 2
        #     finish_mean = 0.
        #     # change timeout so that players have time to finish race
        #     # game.change_timeout(3. + num_segments / 20.)
        #     game.change_timeout(3. + num_segments / 5.)
        #     # beta = 0.  # 0.1 if num_segments < max_segments else 0.001
        #     print(f'{pid:3d} -- Increased number of segments to {num_segments}')

        # discriminator calculate loss and perform backward pass
        winners = one_hot(game.winners() + 1, num_classes=num_players + 1)
        winners = winners.view(trials, -1, *winners.shape[1:]).float().mean(0)
        dloss, dacc = discriminator.train(boards.detach(), winners)
        result['discriminator/loss'] = dloss
        result['discriminator/accuracy'] = dacc

        # -- train generator
        for _ in range(generator_train_steps):
            generated = generator.generate(num_segments, generator_batch_size)
            pred_winners = discriminator.forward(generated)
            gloss, galoss = generator.train(pred_winners, beta)
            result['generator/loss'] = gloss
            if galoss:
                result['generator/aux_loss'] = galoss

        # log data
        for p in range(num_players):
            result[f'game/win_rates/player_{p}'] = winners[:, p + 1].mean().item()
        result['game/invalid'] = winners[:, 0].mean().item()

        # show predefined tracks statistics
        # hidx = winners.size(0) // 2
        # for p in range(num_players):
        #     result[f'predef/win_rates/player_{p}'] = (winners[-15:, p + 1].mean().item() +
        #                                               winners[hidx - 15: hidx, p + 1].mean().item()) / 2.

        # save episode
        if pid == 0 and episode % 100 == 0:
            game.record_episode(os.path.join(run_path, 'videos', f'episode_{episode}'))
            # save boards as images in tensorboard
            # for i, img in enumerate(game.tracks_images(top_n=batch_size + 4 * 6)):
            for i, img in enumerate(game.tracks_images(top_n=batch_size)):
                result[f'game/boards_{i}'] = np.transpose(img, axes=(2, 0, 1))

        # save networks
        if pid == 0 and episode % 500 == 0:
            torch.save(discriminator.network.state_dict(), os.path.join(run_path, f'discriminator_{episode}.pt'))
            torch.save(discriminator.optimizer.state_dict(), os.path.join(run_path, f'discriminator_opt_{episode}.pt'))
            torch.save(generator.network.state_dict(), os.path.join(run_path, f'generator_{episode}.pt'))
            torch.save(generator.optimizer.state_dict(), os.path.join(run_path, f'generator_opt_{episode}.pt'))
            # for i, a in enumerate(agents):
            #     torch.save(a.network.state_dict(), os.path.join(run_path, f'agent_{i}_{episode}.pt'))

        result_queue.put(result)
        result = {}


def log_results(run_path, result_queue: mp.Queue):
    from collections import defaultdict

    print('Starting logging results...')
    summary_writer = SummaryWriter(os.path.join(run_path, 'summary'),
                                   comment='Learning from zero. Discrete generator. Mirrored levels.')

    steps = defaultdict(lambda: 0)
    while True:
        result = result_queue.get()
        for tag, data in result.items():
            if isinstance(data, np.ndarray):
                summary_writer.add_image(tag, data, global_step=steps[tag])
            else:
                summary_writer.add_scalar(tag, data, global_step=steps[tag])
            steps[tag] += 1
        # -----
        if steps['gc-collect'] % 1000 == 0:
            gc.collect()
        steps['gc-collect'] += 1


def main():
    mp.set_start_method('spawn')

    run_path = find_next_run_dir('experiments')
    print(f'Running experiment {run_path}')

    manager = mp.Manager()
    result_queue = manager.Queue(maxsize=1024)

    # # create agents with LSTM policy network
    # cars = [RaceCar(max_speed=60., acceleration=2., angle=40.),
    #         RaceCar(max_speed=60., acceleration=1., angle=80.)]
    # game = Race(timeout=60. * 10., framerate=1. / 20., cars=cars)
    #
    # agents = [PPOAgent(game.actions,
    #                    LSTMPolicy(game.state_shape()[0], game.actions),
    #                    lr=5e-5, discount=0.99, eps=0.1, asynchronous=True)
    #           for _ in range(game.num_players)]
    #
    # del game
    #
    # # load agents if resuming
    # if learned_agents:
    #     for i, a in enumerate(agents):
    #         path = find_latest(learned_agents, 'agent_{}_*.pt'.format(i))
    #         print(f'Resuming agent {i} from path "{path}"')
    #         a.network.load_state_dict(torch.load(path))
    #         a.old_network.load_state_dict(torch.load(path))
    #         a.network.cuda()
    #         a.old_network.cuda()
    #
    # # create discriminator
    # discriminator = RaceWinnerDiscriminator(num_players, lr=1e-5, asynchronous=True)
    #
    # if learned_discriminator:
    #     path = find_latest(learned_discriminator, 'discriminator_[0-9]*.pt')
    #     print(f'Resuming discriminator from path "{path}"')
    #     discriminator.network.load_state_dict(torch.load(path))
    #     discriminator.network.cuda()
    #
    # # create generator
    # generator = RaceTrackGenerator(latent, lr=1e-5, asynchronous=True)
    #
    # if learned_generator:
    #     path = find_latest(learned_generator, 'generator_[0-9]*.pt')
    #     print(f'Resuming generator from path "{path}"')
    #     generator.network.load_state_dict(torch.load(path))
    #     generator.network.cuda()

    # train(generator, discriminator, agents, result_queue, 0, run_path)

    # run a pool of threads
    sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool(num_proc + 1)
    signal.signal(signal.SIGINT, sigint_handler)

    processes = [pool.apply_async(log_results, args=(run_path, result_queue))]
    processes += [pool.apply_async(train, args=(None, None, None, result_queue, 0, run_path))]
    # train(None, None, None, result_queue, 0, run_path)
    # for pid in range(num_proc):
    #     processes.append(pool.apply_async(train, args=(generator, discriminator, agents, result_queue, pid, run_path)))

    try:
        while True:
            for p in processes:
                try:
                    p.get(timeout=1.)
                except mp.TimeoutError:
                    pass
    except KeyboardInterrupt:
        print('Terminating pool...')
        pool.terminate()
    # else:
    #     print('Closing pool...')
    #     pool.close()


if __name__ == '__main__':
    main()
