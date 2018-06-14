import os
import h5py
import queue
import random
import signal
import numpy as np
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils import data
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from game import Race, RaceCar
from agents import PPOAgent
from networks import LSTMPolicy, RaceWinnerDiscriminator
from utils import find_next_run_dir, find_latest, device

# resume = None  # os.path.join('..', 'experiments', '009', 'run-5')
resume = os.path.join('experiments', 'run-2')
num_players = 2
batch_size = 64


def main():
    print('Starting learning discriminator...')
    run_path = find_next_run_dir('experiments')

    discriminator = RaceWinnerDiscriminator(num_players, lr=1e-4, asynchronous=True)
    discriminator.async_optim(optim.Adam(discriminator.network.parameters(), lr=1e-4))
    scheduler = lr_scheduler.ReduceLROnPlateau(discriminator.optimizer, patience=10, factor=0.1,
                                               min_lr=1e-6, verbose=True)
    # discriminator = RaceWinnerDiscriminator(num_players, lr=1e-5)

    if resume:
        path = find_latest(resume, 'discriminator_*.pt')
        print(f'Resuming discriminator from path "{path}"')
        discriminator.network.load_state_dict(torch.load(path))

    with h5py.File('dataset.h5') as file:
        tracks, winners = file['tracks'][:], file['winners'][:]
        winners = np.round(winners * 2.0) / 2.0

        winners = np.concatenate((np.zeros((winners.shape[0], 1), dtype=np.float32), winners), axis=1)

        val_tracks, val_winners = torch.tensor(tracks[-1000:]).to(device), torch.tensor(winners[-1000:]).to(device)
        tracks, winners = torch.tensor(tracks[:-1000]), torch.tensor(winners[:-1000])

    dataset = data.TensorDataset(tracks, winners)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, pin_memory=True)

    summary_writer = SummaryWriter(os.path.join(run_path, 'summary'))

    epoch = 0
    while True:
        print(f'Epoch {epoch:5d}', flush=True)

        step = 0
        for i, (tr, win) in enumerate(dataloader):
            step += tr.size(0)
            # print(f'\r[{step:5d}/{len(dataset):5d}]', end='')
            dloss, dacc = discriminator.train(tr.to(device), win.to(device))
            summary_writer.add_scalar('summary/discriminator_loss', dloss, global_step=epoch * len(dataset) + step)
            summary_writer.add_scalar('summary/discriminator_accuracy', dacc, global_step=epoch * len(dataset) + step)

        if epoch % 10 == 0:
            torch.save(discriminator.network.state_dict(), os.path.join(run_path, f'discriminator_{epoch}.pt'))

        with torch.no_grad():
            val_loss, val_acc = discriminator.loss(val_tracks, val_winners)
            summary_writer.add_scalar('summary/validation/discriminator_loss', val_loss, global_step=epoch)
            summary_writer.add_scalar('summary/validation/discriminator_accuracy', val_acc, global_step=epoch)
            scheduler.step(val_loss, epoch=epoch)

        epoch += 1


if __name__ == '__main__':
    main()
