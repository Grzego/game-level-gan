import math
import torch
from torch.utils import cpp_extension as ext

from utils import device
from .environment import MultiEnvironment


class RaceCar(object):
    def __init__(self, max_speed, acceleration, angle):
        # max_speed in km/h
        # max_acceleration in m/s^2
        # max_angle in degrees/s

        # converting to units/s, units/s^2 and rads/s
        self.max_speed = max_speed * 100. / 3600.
        self.acceleration = acceleration * 0.1
        self.angle = angle * math.pi / 180.


class Race(MultiEnvironment):
    IMPL_BOOST = 0
    IMPL_GPU = 1
    IMPL_CPP = 2

    def __init__(self, timeout, cars: [RaceCar], observation_size=18, max_distance=10., framerate=1. / 30.,
                 log_history=True, device=device):
        # 1unit = 10m
        # TODO: try continuous actions
        self.device = device
        self.timeout = timeout
        self.framerate = framerate
        self.num_players = len(cars)
        self.cars_max_speed = torch.tensor([car.max_speed for car in cars], dtype=torch.float32, device=device)
        self.cars_acceleration = torch.tensor([car.acceleration for car in cars], dtype=torch.float32, device=device)
        self.cars_angle = torch.tensor([car.angle for car in cars], dtype=torch.float32, device=device)
        self.num_tracks = None
        self.segments = None
        self.positions = None
        self.directions = None
        self.speeds = None
        self.alive = None
        self.finishes = None
        self.scores = None
        self.valid = None
        self.steps = 0
        self.steps_limit = int(timeout // framerate)
        self.bounds = None
        self.left_vecs = None
        self.right_vecs = None
        self.reward_bound = None
        self.action_speed = torch.tensor([0.,  # noop
                                          1.,  # forward
                                          -3.,  # backward
                                          0.,  # right
                                          1.,  # forward-right
                                          -3.,  # backward-right
                                          0.,  # left
                                          1.,  # forward-left
                                          -3.,  # backward-left
                                          ], device=device)
        self.action_dirs = torch.tensor([0.,  # noop
                                         0.,  # forward
                                         0.,  # backward
                                         1.,  # right
                                         1.,  # forward-right
                                         1.,  # backward-right
                                         -1.,  # left
                                         -1.,  # forward-left
                                         -1.,  # backward-left
                                         ], device=device)
        self.observation_size = observation_size
        self.max_distance = max_distance
        self.negative_reward = -0.01
        self.log_history = log_history
        self.history = []  # keeps posision and direction for players on first board
        self.record_id = 0

        self.line_bounds = None
        self.game_handle = None

        self._impl_version = Race.IMPL_CPP

        try:
            self.game_helpers = ext.load('game_helpers',
                                         sources=['game/game_helpers.cpp'],
                                         extra_cflags=['-DNDEBUG', '-O3', '-fopenmp'],
                                         extra_ldflags=['-lpthread'])
        except RuntimeError as e:
            import codecs
            print(codecs.decode(e.args[0], 'unicode-escape'))
            print('*****************************************************')
            print('!!! Fallbacking to default PyTorch implementation !!!')
            print('*****************************************************')
            self._impl_version = Race.IMPL_GPU

    def state_shape(self):
        return self.observation_size + 2,  # +1 for speed, (+1 for progress -- REMOVED)

    def players_layer_shape(self):
        pass

    def record(self, board):
        self.record_id = board

    def change_timeout(self, timeout):
        self.timeout = timeout
        self.steps_limit = int(timeout // self.framerate)

    def reset(self, tracks):
        """
        tracks - [num_boards, segments, (arc, width)]

        where:
            arc is in range (-1, 1) meaning next segment angle from (-90deg, 90deg)
            width is in range (0, 1) meaning width from (0.5, 2.0)

        # TODO: test different settings
        """
        length = 0.2  # each segment is 2m long
        min_width, max_width = 0.5, 2.0

        with torch.no_grad():
            self.steps = 0
            self.num_tracks = tracks.size(0)
            self.history = []

            # add sentinels to tracks (0 in front and 0 in back of track for additional segment)
            num_boards = tracks.size(0)
            tracks = torch.cat((torch.zeros((num_boards, 1, 2), device=self.device), tracks,
                                torch.zeros((num_boards, 1, 2), device=self.device)),
                               dim=1)

            arcsum = math.radians(8.) * torch.cumsum(tracks[:, :, :1],
                                                     dim=1)  # cumsum over angles and conversion to radians
            segment_vecs = torch.cat((torch.sin(arcsum), torch.cos(arcsum)), dim=2) * length

            perp_vecs = segment_vecs.clone()
            perp_vecs[:, :, 0], perp_vecs[:, :, 1] = segment_vecs[:, :, 1], -segment_vecs[:, :, 0]

            right_vecs = perp_vecs[:, 1:, :] + perp_vecs[:, :-1, :]
            right_vecs /= right_vecs.norm(p=2., dim=-1, keepdim=True)
            right_vecs *= min_width + (max_width - min_width) * tracks[:, :-1, 1:]
            right_vecs = torch.cat((torch.zeros((num_boards, 1, 2), device=self.device), right_vecs), dim=1)
            right_vecs[:, 0, 0] = min_width
            left_vecs = -right_vecs

            segments = torch.cumsum(segment_vecs, dim=1)
            segments[:, 1:, :] = segments[:, :-1, :].clone()
            segments[:, 0, :] = 0.
            right_vecs = segments + right_vecs
            left_vecs = segments + left_vecs

            # -- for C++ version
            self.right_vecs = right_vecs
            self.left_vecs = left_vecs

            self.segments = segments

            right_bounds = torch.cat((right_vecs[:, :-1, :], right_vecs[:, 1:, :]), dim=-1)
            left_bounds = torch.cat((left_vecs[:, :-1, :], left_vecs[:, 1:, :]), dim=-1)
            start_bounds = torch.cat((left_vecs[:, :1, :], right_vecs[:, :1, :]), dim=-1)
            reward_line = torch.cat((left_vecs[:, -1:, :], right_vecs[:, -1:, :]), dim=-1)
            self.reward_bound = reward_line.unsqueeze(1).repeat(1, self.num_players, 1, 1) \
                .view(-1, *reward_line.shape[-2:])
            bounds = torch.cat((right_bounds, left_bounds, start_bounds), dim=1)  # [num_boards * num_players, num_segments, 4]
            self.bounds = bounds.unsqueeze(1).repeat(1, self.num_players, 1, 1).view(-1, *bounds.shape[-2:])

            line_bounds = torch.cat((right_vecs.flip(1), left_vecs), dim=1)
            self.line_bounds = line_bounds.unsqueeze(1).repeat(1, self.num_players, 1, 1)\
                .view(-1, *line_bounds.shape[-2:]).cpu()

            self.left_bounds = left_bounds
            self.right_bounds = right_bounds

            self.positions = torch.zeros((num_boards, self.num_players, 2), device=self.device)
            self.positions[:, :, 1] = 0.1  # 1m after a start
            self.directions = torch.zeros((num_boards, self.num_players, 2), device=self.device)
            self.directions[:, :, 1] = 1.  # direction vectors should have length = 1.
            self.speeds = torch.zeros((num_boards, self.num_players), device=self.device)
            self.alive = torch.ones((num_boards, self.num_players), dtype=torch.uint8, device=self.device)
            self.scores = torch.empty((num_boards, self.num_players), dtype=torch.int32, device=self.device)
            self.scores.zero_()
            self.finishes = torch.zeros((num_boards, self.num_players), dtype=torch.uint8, device=self.device)

            # -- boost version
            if self._impl_version == Race.IMPL_BOOST:
                valid = torch.empty(num_boards, dtype=torch.uint8)
                self.game_helpers.is_valid(line_bounds.cpu(), valid)
                valid = valid.to(device)

            # -- brute-force version on GPU
            elif self._impl_version == Race.IMPL_GPU:
                valid = self._is_correct(torch.cat((bounds, reward_line), dim=1))

            # -- fully in C++
            elif self._impl_version == Race.IMPL_CPP:
                self.game_handle = self.game_helpers.Game(self.left_vecs.cpu(), self.right_vecs.cpu(), self.num_players)
                valid = self.game_handle.validate_tracks().to(device)

            self.valid = valid.view(-1, 1).repeat(1, self.num_players).view(-1).contiguous()
            any_valid = valid.sum().item() > 0

            return self.step(torch.zeros((self.num_players, num_boards),
                                         dtype=torch.int64, device=self.device))[0], any_valid

    @staticmethod
    def _segment_collisions(segments, tests, special=False, return_orientations=False):
        """
        Tests collision between every S and P for every N.
        Returns True/False wherther collision occured (shape [N, S, P])

        special - returns separate collision masks [collisions, collision with start point on segment]

        segments = [N, S, 4]
        tests = [N, P, 4]

        N - usually batch size or number of boards
        S - number of segments
        P - number of players
        """
        n, s, p = segments.size(0), segments.size(1), tests.size(1)

        def _orientation(p, q, r):
            """
            Returns orientation of r point with respect to segment p -> q
            (-1, 0, 1) - (widdershins - counterclockwise, colinear, clockwise)
            """
            p, q, r = p.unsqueeze(2), q.unsqueeze(2), r.unsqueeze(1)
            qp = q - p
            rq = r - q
            return torch.sign(qp[:, :, :, 1] * rq[:, :, :, 0] - qp[:, :, :, 0] * rq[:, :, :, 1])

        def _on_segment(p, q, r):
            """
            Checks whether point r lies on p -> q segment
            """
            p, q, r = p.unsqueeze(2), q.unsqueeze(2), r.unsqueeze(1)
            return torch.prod((r <= torch.max(p, q)) & (r >= torch.min(p, q)), dim=-1, dtype=torch.uint8)
            # return ((r <= torch.max(p, q)) & (r >= torch.min(p, q))).all(-1)

        p1, q1 = segments[:, :, :2], segments[:, :, 2:]
        p2, q2 = tests[:, :, :2], tests[:, :, 2:]

        o1 = _orientation(p1, q1, p2).view(n, -1)  # [N, S * P]
        o2 = _orientation(p1, q1, q2).view(n, -1)  # [N, S * P]
        o3 = _orientation(p2, q2, p1).permute(0, 2, 1).contiguous().view(n, -1)  # [N, S * P] - permute to match cases
        o4 = _orientation(p2, q2, q1).permute(0, 2, 1).contiguous().view(n, -1)  # [N, S * P]

        # general case
        res = (o1 != o2) & (o3 != o4)

        if return_orientations:
            # return o1.view(n, s, p), o2.view(n, s, p), o3.view(n, s, p), o4.view(n, s, p)
            return o1, o2, o3, o4

        # special cases
        spec = (o1 == 0.) & _on_segment(p1, q1, p2).view(n, -1)
        res |= (o2 == 0.) & _on_segment(p1, q1, q2).view(n, -1)
        res |= (o3 == 0.) & _on_segment(p2, q2, p1).permute(0, 2, 1).contiguous().view(n, -1)
        res |= (o4 == 0.) & _on_segment(p2, q2, q1).permute(0, 2, 1).contiguous().view(n, -1)

        return (res | spec).view(n, s, p) if not special else (res.view(n, s, p), spec.view(n, s, p))

    @staticmethod
    def _smallest_distance(segments, directions):
        """
        Returns smallest distances to any segments in given `directions`.

        segments = [N, S, 4]
        directions = [N, D, 4]

        N - usually batch size or number of boards
        S - number of segments
        D - number of distinct directions

        Returns tensor with shape [N, D]
        """
        n, d, s = directions.size(0), directions.size(1), segments.size(1)

        # get collisions mask
        far_dirs = directions.clone()
        far_dirs[:, :, 2:] = directions[:, :, :2] + 1000. * directions[:, :, 2:]
        # _segment_collisions (with removing starting point)
        collisions, start_on_segment = Race._segment_collisions(segments, far_dirs, special=True)
        collisions &= ~start_on_segment

        p, q = segments[:, :, None, :2], segments[:, :, None, 2:]
        s, d = directions[:, None, :, :2], directions[:, None, :, 2:]

        qpc = q - p  # [N, S, 1, 2]
        spc = p - s  # [N, S, D, 2]

        dists = spc[:, :, :, 1] * qpc[:, :, :, 0] - spc[:, :, :, 0] * qpc[:, :, :, 1]  # [N, S, D]
        denom = d[:, :, :, 1] * qpc[:, :, :, 0] - d[:, :, :, 0] * qpc[:, :, :, 1]  # [N, S, D]

        dists[start_on_segment] = 0.
        dists[collisions] /= denom[collisions]
        dists[~(collisions | start_on_segment)] = float('inf')
        dists[dists < 0.] = float('inf')

        return torch.min(dists, dim=1)[0]

    @staticmethod
    def _rotate_vecs(vectors, angles):
        """
        Rotates vectors to the left by given angle.

        vectors = [N, 2]
        angles = [N]
        """
        angles = angles.view(-1, 1).repeat(1, 4).view(-1, 2, 2)
        angles[:, 0, 0].cos_()
        angles[:, 0, 1].sin_().neg_()
        angles[:, 1, 0].sin_()
        angles[:, 1, 1].cos_()

        return torch.matmul(vectors.unsqueeze(1), angles).view(-1, 2)

    def _is_correct(self, bounds):
        """
        Returns which boards are correct.
        output_shape = [num_boards]

        bounds = [num_boards, num_segments, 4]
        """
        o1, o2, o3, o4 = self._segment_collisions(bounds, bounds, return_orientations=True)
        return ~torch.max((o1 * o2 < 0) & (o3 * o4 < 0), dim=-1)[0]

    def iterate_valid(self, agents):
        valid = self.valid.view(-1, self.num_players)[:, 0].tolist()
        return ((i, a) for i, a in enumerate(agents) if valid[i])

    def step(self, actions):
        """
        actions - torch.Tensor with size [num_players, num_boards]
        """
        with torch.no_grad():
            # TODO: introduce friction/drag?
            drag = 0.05
            actions = actions.t().contiguous()

            self.steps += 1
            num_boards, num_seg = actions.size(0), self.bounds.size(2)
            # num_boards, num_seg = actions.size(0), self.segments.size(1)

            if self.alive.sum().item() == 0:  # nobody is alive
                states = torch.zeros((self.num_players, num_boards, self.observation_size + 1), device=self.device)
                rewards = (1. - self.finishes.float()) * self.negative_reward
                return states, rewards.t()

            # if player won/died ignore it's action
            actions[~self.alive | ~self.valid.view(self.alive.shape)] = 0  # noop

            #  choose move vector (thats car and action dependent)
            dir_flag = self.action_dirs[actions.view(-1)]  # [num_boards * num_players]
            angles = self.framerate * dir_flag * self.cars_angle.repeat(num_boards).view(-1)  # [num_boards * num_players]
            new_dirs = self._rotate_vecs(self.directions.view(-1, 2), angles).view(-1, self.num_players, 2)

            speed_flag = self.action_speed[actions.view(-1)].view(num_boards, -1)  # [num_boards, num_players]
            speed = self.speeds + self.framerate * speed_flag * self.cars_acceleration[None, :]
            # new_speed = torch.min(self.cars_max_speed, torch.max(self.cars_max_speed.neg(), speed))
            new_speed = torch.min(self.cars_max_speed, speed.clamp(min=0.))  # with break and no backward movements
            is_moving = torch.abs(new_speed.view(-1)) > 1e-7  # [num_boards * num_players]

            new_pos = self.positions + new_dirs * new_speed[:, :, None]

            finish_dist = torch.norm(new_pos[:, :, None, :] - self.segments[:, None, :, :], dim=-1)
            finish_dist_score = finish_dist.argmin(-1, keepdim=True)
            finish_dist = finish_dist_score.float() / (num_seg - 1)
            # finish_dist = finish_dist.argmin(-1, keepdim=True).float() / (2 * num_seg)

            # pick boards with alive players
            update_mask = (self.alive.view(-1) & is_moving & self.valid).nonzero().squeeze(-1)

            rewards = torch.zeros((num_boards * self.num_players), device=self.device)
            rewards[~self.finishes.view(-1)] = self.negative_reward  # small negative reward over time for not finished

            if update_mask.numel() > 0:  # anyone moved
                # check collisions

                # -- boost & GPU versions
                if self._impl_version == Race.IMPL_BOOST or self._impl_version == Race.IMPL_GPU:
                    paths = torch.cat((self.positions, new_pos), dim=-1).view(-1, 1, 4)  # [num_boards * num_players, 1, 4]

                # -- C++ version
                elif self._impl_version == Race.IMPL_CPP:
                    paths = torch.cat((self.positions, new_pos), dim=-1).view(-1, 4)  # [num_boards * num_players, 1, 4]

                # -- boost version
                if self._impl_version == Race.IMPL_BOOST:
                    is_dead = torch.empty(update_mask.numel(), 1, dtype=torch.uint8)
                    self.game_helpers.collision(self.line_bounds[update_mask],
                                                paths[update_mask].cpu(),
                                                is_dead)
                    is_dead = is_dead.to(device).squeeze(-1)

                # -- brute-force version on GPU
                elif self._impl_version == Race.IMPL_GPU:
                    seg_col = self._segment_collisions(self.bounds[update_mask], paths[update_mask])
                    is_dead = seg_col.squeeze(dim=-1).max(dim=1)[0]

                # -- fully in C++
                elif self._impl_version == Race.IMPL_CPP:
                    is_dead_cpu, is_done_cpu = self.game_handle.update_players(update_mask.cpu(), paths[update_mask].cpu())
                    is_dead, is_done = is_dead_cpu.to(device), is_done_cpu.to(device)

                self.alive.view(-1)[update_mask] &= ~is_dead

                # check for reward
                # +1 if finished
                # -1 if died
                # small negative otherwise to encourage finishing race faster

                # -- boost version
                if self._impl_version == Race.IMPL_BOOST:
                    is_done = torch.empty(update_mask.numel(), 1, dtype=torch.uint8)
                    self.game_helpers.collision(self.reward_bound[update_mask].cpu().view(-1, 2, 2),
                                                paths[update_mask].cpu(),
                                                is_done)
                    is_done = is_done.to(device).squeeze(-1)

                # -- brute-force version on GPU
                elif self._impl_version == Race.IMPL_GPU:
                    finish = self._segment_collisions(self.reward_bound[update_mask], paths[update_mask])
                    is_done = torch.max(finish.view(update_mask.size(0), -1), dim=-1)[0]

                rewards[update_mask] += is_done.float() - is_dead.float()
                self.alive.view(-1)[update_mask] &= ~is_done
                self.finishes.view(-1)[update_mask] |= is_done

                # dead_or_done = (is_dead | is_done).int()
                is_dead_int = is_dead.int()
                is_done_int = is_done.int()

                self.scores.view(-1)[update_mask] = self.scores.view(-1)[update_mask] * (1 - is_dead_int) \
                                                    + is_dead_int * (finish_dist_score.view(-1)[update_mask].int()
                                                                     + self.steps_limit + 1)

                self.scores.view(-1)[update_mask] = self.scores.view(-1)[update_mask] * (1 - is_done_int) \
                                                    + is_done_int * self.steps
                # stop finished players
                new_speed[~self.alive] = 0.

            # update all player variables
            drags = 1. - (1. - (speed_flag != 0.).float()) * drag

            self.directions = new_dirs
            self.speeds = new_speed * drags.view(new_speed.shape)
            self.positions = new_pos

            # return states (state is just distances in few directions)
            states = torch.zeros((num_boards * self.num_players, self.observation_size), device=self.device)
            alive_mask = self.alive.view(-1).nonzero().squeeze(-1)
            if alive_mask.numel() > 0:
                obs_angles = torch.linspace(-math.pi, math.pi * (1. - 2. / self.observation_size),
                                            self.observation_size).to(self.device).view(1, -1)
                obs_angles = obs_angles.repeat(num_boards * self.num_players, 1).view(-1)
                rot_dirs = new_dirs.view(-1, 1, 2).repeat(1, self.observation_size, 1).view(-1, 2)
                obs_dirs = self._rotate_vecs(rot_dirs, obs_angles)
                obs_dirs = obs_dirs.view(-1, self.observation_size,
                                         2)  # [num_boards * num_players, observation_size, 2]
                obs_segm = new_pos.view(-1, 1, 2).repeat(1, self.observation_size, 1)
                obs_segm = torch.cat((obs_segm, obs_dirs), dim=-1)

                # -- boost version
                if self._impl_version == Race.IMPL_BOOST:
                    alive_states = torch.empty(alive_mask.numel(), self.observation_size, dtype=torch.float)
                    self.game_helpers.smallest_distance(self.line_bounds[alive_mask].cpu(),
                                                        obs_segm[alive_mask].cpu(),
                                                        alive_states)
                    alive_states = alive_states.to(device)

                # -- brute-force version on GPU
                elif self._impl_version == Race.IMPL_GPU:
                    alive_states = self._smallest_distance(self.bounds[alive_mask], obs_segm[alive_mask])

                # -- fully in C++
                elif self._impl_version == Race.IMPL_CPP:
                    alive_states = self.game_handle.smallest_distance(alive_mask.cpu(),
                                                                      obs_segm[alive_mask].cpu()).to(device)

                states[alive_mask] = alive_states.clamp(max=self.max_distance) / self.max_distance

            # record history
            if self.log_history:
                self.history.append((self.positions[self.record_id].tolist(), self.directions[self.record_id].tolist(),
                                     actions[self.record_id].tolist(), self.alive[self.record_id].tolist()))

            states = torch.cat((states.view(num_boards, self.num_players, -1),
                                self.speeds[:, :, None] / self.cars_max_speed[None, :, None],
                                finish_dist),
                               dim=-1)
            return states.permute(1, 0, 2), rewards.view(num_boards, -1).t()

    def finished(self):
        #      timeout was reached           or all players are crushed
        return self.steps > self.steps_limit or not self.alive.any()  # torch.sum(self.alive).item() < 0.1

    def winners(self):
        """
        Rules:
          if any player finished, faster one is a winner
          if both players are dead, one that survived the longest is a winner

          -1 means board was invalid
        """
        with torch.no_grad():
            winner = torch.zeros((self.finishes.size(0),), dtype=torch.long, device=self.device)

            anyone_finished = self.finishes.sum(-1) > 0
            if torch.sum(anyone_finished) > 0:
                finished = self.scores[anyone_finished].clone()
                finished[~self.finishes[anyone_finished]] = self.steps_limit + 1

                winner[anyone_finished] = torch.argmin(finished, dim=-1)
                if torch.sum(~anyone_finished) > 0:
                    winner[~anyone_finished] = torch.argmax(self.scores[~anyone_finished], dim=-1)
            else:
                winner = torch.argmax(self.scores, dim=-1)

            winner[~self.valid.view(-1, self.num_players)[:, 0]] = -1
            return winner

    def record_episode(self, filename: str):
        """
        Saves a movie with gameplay on first track to a file.
        Requires OpenCV and movie-py.

        filename - without extension

        This method is rather inefficient and should be rarely used.
        """
        if not self.log_history:
            print('Logging of history is tuned off.')
            return

        import os
        import cv2
        # import moviepy.editor as mpy
        import numpy as np

        width, height = 640, 480
        scale = 150.  # px == 1unit
        board = self.record_id * self.num_players

        record = 255 * np.ones((self.num_players, len(self.history), height, width, 3), dtype=np.uint8)

        # right_bounds = torch.cat((self.right_vecs[:, :-1, :], self.right_vecs[:, 1:, :]), dim=-1)
        # left_bounds = torch.cat((self.left_vecs[:, :-1, :], self.left_vecs[:, 1:, :]), dim=-1)
        # start_bounds = torch.cat((self.left_vecs[:, :1, :], self.right_vecs[:, :1, :]), dim=-1)
        # bounds = torch.cat((right_bounds, left_bounds, start_bounds), dim=1)
        # bounds = bounds.unsqueeze(1).repeat(1, self.num_players, 1, 1).view(-1, *bounds.shape[-2:])
        # reward_line = torch.cat((self.left_vecs[:, -1:, :], self.right_vecs[:, -1:, :]), dim=-1)
        # reward_bound = reward_line.unsqueeze(1).repeat(1, self.num_players, 1, 1).view(-1, *reward_line.shape[-2:])

        bounds = self.bounds
        reward_bound = self.reward_bound
        cut = None

        for frame, (positions, directions, actions, alive) in enumerate(self.history):
            if not any(alive):
                cut = frame
                break

            for player, (position, direction, action) in enumerate(zip(positions, directions, actions)):
                # center on player
                px, py = position
                px *= scale
                py *= scale
                px = width // 2 - px
                py = height // 2 - py
                # draw every segment of first track
                for x1, y1, x2, y2 in bounds[board, :, :]:
                    x1, y1, x2, y2 = map(int, (x1 * scale + px, y1 * scale + py,
                                               x2 * scale + px, y2 * scale + py))
                    cv2.line(record[player, frame], (x1, height - y1), (x2, height - y2), (0, 0, 0, 0),
                             thickness=3, lineType=cv2.LINE_AA)
                # distances
                obs_angles = torch.linspace(-math.pi, math.pi * (1. - 2. / self.observation_size),
                                            self.observation_size).to(self.device)
                rot_dirs = torch.tensor(direction, device=self.device).view(1, 2).repeat(self.observation_size, 1)
                obs_dirs = self._rotate_vecs(rot_dirs, obs_angles)
                obs_dirs = obs_dirs.view(self.observation_size, 2)
                obs_segm = torch.tensor(position, device=self.device).view(1, 2).repeat(self.observation_size, 1)
                obs_segm = torch.cat((obs_segm, obs_dirs), dim=-1)
                #
                alive_states = self._smallest_distance(bounds[board: board + 1, :, :], obs_segm[None, :, :])
                alive_states = alive_states.clamp(max=self.max_distance).squeeze(0)

                mx, my = width // 2, height // 2
                for d, dist in zip(obs_dirs, alive_states):
                    d *= scale * dist
                    x, y = d
                    cv2.line(record[player, frame], (mx, height - my),
                             (int(x.item()) + mx, height - int(y.item()) - my),
                             (0, 170, 0, 0), thickness=1, lineType=cv2.LINE_AA)

                # player position
                fx1, fy1, fx2, fy2 = reward_bound[board, 0, :]
                fx1, fy1, fx2, fy2 = map(int, (fx1 * scale + px, fy1 * scale + py,
                                               fx2 * scale + px, fy2 * scale + py))
                cv2.line(record[player, frame], (fx1, height - fy1), (fx2, height - fy2), (170, 0, 0, 0),
                         thickness=3, lineType=cv2.LINE_AA)

                dx, dy = direction
                dx *= scale * 0.1
                dy *= scale * 0.1
                px, py = width // 2, height // 2
                x1, y1, x2, y2 = map(int, (px, py, px + dx, py + dy))
                cv2.line(record[player, frame], (x1, height - y1), (x2, height - y2), (100, 149, 237, 0),
                         thickness=4, lineType=cv2.LINE_AA)

                if action > 0:
                    offset = [(0, 0), (0, -15), (0, 15),
                              (15, 0), (10, -10), (-10, 10),
                              (-15, 0), (-10, -10), (10, 10)][action]
                    offset = 2 * offset[0] + 40, height - 40 + 2 * offset[1]

                    cv2.arrowedLine(record[player, frame], (40, height - 40), offset,
                                    (0, 0, 0, 0), thickness=3, line_type=cv2.LINE_AA)

        record = np.concatenate(list(record), axis=-2)[:cut]
        record = record[..., ::-1]  # because OpenCV...

        basepath, _ = os.path.split(filename)
        if basepath:
            os.makedirs(basepath, exist_ok=True)
        # clip = mpy.ImageSequenceClip(list(record), fps=int(1. / self.framerate))
        # clip.write_videofile(filename + '.mp4', audio=False, verbose=False)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        clip = cv2.VideoWriter(filename + '.mp4', fourcc, 1. / self.framerate, (width * self.num_players, height))
        print()
        print('Saving clip: "{}"'.format(filename + '.mp4'))
        for i, frame in enumerate(record):
            print('\r[{:5d}/{:5d}]'.format(i + 1, len(record)), end='')
            clip.write(frame)
        print('... done.')
        clip.release()
        del clip

    def tracks_images(self, top_n=3):
        import cv2
        import numpy as np

        size = 256
        imgs = 255 * np.ones((top_n, size, size, 3), dtype=np.uint8)

        # right_bounds = torch.cat((self.right_vecs[:, :-1, :], self.right_vecs[:, 1:, :]), dim=-1)
        # left_bounds = torch.cat((self.left_vecs[:, :-1, :], self.left_vecs[:, 1:, :]), dim=-1)
        # start_bounds = torch.cat((self.left_vecs[:, :1, :], self.right_vecs[:, :1, :]), dim=-1)
        # bounds = torch.cat((right_bounds, left_bounds, start_bounds), dim=1)
        # bounds = bounds.unsqueeze(1).repeat(1, self.num_players, 1, 1).view(-1, *bounds.shape[-2:])
        # reward_line = torch.cat((self.left_vecs[:, -1:, :], self.right_vecs[:, -1:, :]), dim=-1)
        # reward_bound = reward_line.unsqueeze(1).repeat(1, self.num_players, 1, 1).view(-1, *reward_line.shape[-2:])

        bounds = self.bounds
        reward_bound = self.reward_bound

        for i in range(top_n):
            mins, _ = torch.min(bounds[i * self.num_players, :, :].view(-1, 2), dim=0)
            maxs, _ = torch.max(bounds[i * self.num_players, :, :].view(-1, 2), dim=0)
            longer = torch.max(maxs - mins).item()
            shift = 0.5 * (1. - (maxs - mins) / longer)
            minx, miny = mins.tolist()
            shiftx, shifty = shift.tolist()

            def _move_x(x):
                return int(0.05 * size + 0.9 * size * ((x - minx) / longer + shiftx))

            def _move_y(y):
                return size - int(0.05 * size + 0.9 * size * ((y - miny) / longer + shifty))

            # draw every segment of first track
            for x1, y1, x2, y2 in bounds[i * self.num_players, :, :]:
                cv2.line(imgs[i], (_move_x(x1), _move_y(y1)), (_move_x(x2), _move_y(y2)), (0, 0, 0, 0),
                         thickness=2, lineType=cv2.LINE_AA)

            fx1, fy1, fx2, fy2 = reward_bound[i * self.num_players, 0, :]
            cv2.line(imgs[i], (_move_x(fx1), _move_y(fy1)), (_move_x(fx2), _move_y(fy2)), (170, 0, 0, 0),
                     thickness=3, lineType=cv2.LINE_AA)

        return imgs

    def prettier_tracks(self, top_n=3, size=1024, pad=0.05):
        import cv2
        import numpy as np

        imgs = 255 * np.ones((top_n, size, size, 4), dtype=np.uint8)
        imgs[:, :, :, 3] = 0

        for i in range(top_n):
            mins, _ = torch.min(self.bounds[i * self.num_players, :, :].view(-1, 2), dim=0)
            maxs, _ = torch.max(self.bounds[i * self.num_players, :, :].view(-1, 2), dim=0)
            # mins, _ = torch.tensor([-20., -20.]), 0
            # maxs, _ = torch.tensor([ 20.,  20.]), 0

            longer = torch.max(maxs - mins).item()
            shift = 0.5 * (1. - (maxs - mins) / longer)
            minx, miny = mins.tolist()
            shiftx, shifty = shift.tolist()

            def _move_x(x):
                return int(pad * size + (1. - 2. * pad) * size * ((x - minx) / longer + shiftx))

            def _move_y(y):
                return size - int(pad * size + (1. - 2. * pad) * size * ((y - miny) / longer + shifty))

            def _move_xy(xy):
                return (_move_x(xy[0]), _move_y(xy[1])), (_move_x(xy[2]), _move_y(xy[3]))

            for j, (l, p) in enumerate(zip(self.left_bounds[i, ...], self.right_bounds[i, ...])):
                l_1, l_2 = _move_xy(l)
                p_1, p_2 = _move_xy(p)

                cv2.fillConvexPoly(imgs[i], np.array([l_1, l_2, p_2, p_1]),
                                   (70, 70, 70, 255) if (j // 5) % 2 == 0 else (50, 50, 50, 255),
                                   lineType=cv2.LINE_AA)

            # draw every segment of first track
            for x1, y1, x2, y2 in self.bounds[i * self.num_players, :, :]:
                cv2.line(imgs[i], (_move_x(x1), _move_y(y1)), (_move_x(x2), _move_y(y2)), (5, 5, 5, 255),
                         thickness=1, lineType=cv2.LINE_AA)

            f1, f2 = _move_xy(self.reward_bound[i * self.num_players, 0, :])
            fl, fr = np.array(f1), np.array(f2)
            perp = (np.array([fl[1] - fr[1], -(fl[0] - fr[0])]) * 0.2).astype(np.int32)

            x_steps, y_steps = 2, 7
            xx_step, yy_step = perp / x_steps, (fr - fl) / y_steps

            for xx in range(x_steps):
                for yy in range(y_steps):
                    p = fl + xx * xx_step + yy * yy_step

                    cv2.fillConvexPoly(imgs[i], np.stack((p, p + xx_step, p + yy_step + xx_step, p + yy_step)).astype(np.int32),
                                       (20, 20, 20, 255) if (xx % 2 == 0) == (yy % 2 == 0) else (210, 210, 210, 255),
                                       lineType=cv2.LINE_AA)

            # cv2.line(imgs[i], (_move_x(fx1), _move_y(fy1)), (_move_x(fx2), _move_y(fy2)), (0, 0, 145, 255),
            #          thickness=1, lineType=cv2.LINE_AA)

        return imgs

    def prettier_tracks_svg(self, top_n=3, size=1024, pad=0.05):
        import numpy as np
        import svgwrite as svg

        imgs = [svg.Drawing(shape_rendering="crispEdges") for _ in range(top_n)]

        for i in range(top_n):
            mins, _ = torch.min(self.bounds[i * self.num_players, :, :].view(-1, 2), dim=0)
            maxs, _ = torch.max(self.bounds[i * self.num_players, :, :].view(-1, 2), dim=0)
            # mins, _ = torch.tensor([-20., -20.]), 0
            # maxs, _ = torch.tensor([ 20.,  20.]), 0

            longer = torch.max(maxs - mins).item()
            shift = 0.5 * (1. - (maxs - mins) / longer)
            minx, miny = mins.tolist()
            shiftx, shifty = shift.tolist()

            def _move_x(x):
                return float(pad * size + (1. - 2. * pad) * size * ((x - minx) / longer + shiftx))

            def _move_y(y):
                return float(size - (pad * size + (1. - 2. * pad) * size * ((y - miny) / longer + shifty)))

            def _move_xy(xy):
                return (_move_x(xy[0]), _move_y(xy[1])), (_move_x(xy[2]), _move_y(xy[3]))

            def _norm(v):
                return v / np.sqrt(np.sum(v**2.))

            def _adjust(points):
                l1, l2, p2, p1 = np.array(points)
                r = p2 - l2
                u = l2 - l1
                l1 += _norm(-r - u) * 0.01 * size
                l2 += _norm(-r + u) * 0.01 * size
                p2 += _norm( r + u) * 0.01 * size
                p1 += _norm( r - u) * 0.01 * size
                return [x.tolist() for x in (l1, l2, p2, p1)]

            for j, (l, p) in enumerate(zip(self.left_bounds[i, ...], self.right_bounds[i, ...])):
                l_1, l_2 = _move_xy(l)
                p_1, p_2 = _move_xy(p)

                imgs[i].add(imgs[i].polygon(points=_adjust([l_1, l_2, p_2, p_1]),
                                            fill=svg.rgb(70, 70, 70) if (j // 5) % 2 == 0 else svg.rgb(50, 50, 50)))

            # draw every segment of first track
            # for x1, y1, x2, y2 in self.bounds[i * self.num_players, :, :]:
            #     imgs[i].add(imgs[i].line((_move_x(x1), _move_y(y1)), (_move_x(x2), _move_y(y2)),
            #                 style='stroke:rgb(5, 5, 5, 255), stroke-width:1'))

            f1, f2 = _move_xy(self.reward_bound[i * self.num_players, 0, :])
            fl, fr = np.array(f1), np.array(f2)
            r = _norm(fr - fl) * 0.01 * size
            fl -= r
            fr += r
            perp = (np.array([fl[1] - fr[1], -(fl[0] - fr[0])]) * 0.22).astype(np.int32)

            x_steps, y_steps = 2, 7
            xx_step, yy_step = perp / x_steps, (fr - fl) / y_steps

            for xx in range(x_steps):
                for yy in range(y_steps):
                    p = fl + xx * xx_step + yy * yy_step
                    points = np.stack((p, p + xx_step, p + yy_step + xx_step, p + yy_step)).tolist()
                    imgs[i].add(imgs[i].polygon(points=points,
                                                fill=svg.rgb(20, 20, 20) if (xx % 2 == 0) == (yy % 2 == 0) else svg.rgb(210, 210, 210)))

            # cv2.line(imgs[i], (_move_x(fx1), _move_y(fy1)), (_move_x(fx2), _move_y(fy2)), (0, 0, 145, 255),
            #          thickness=1, lineType=cv2.LINE_AA)

        return imgs

    def play(self):
        """
        Draws first board and first player in interactive mode.
        """
        import time
        import pyglet as pgl
        from pyglet.window import key

        window = pgl.window.Window(width=640, height=480, caption='Race')
        actions = {key.UP: False, key.DOWN: False,
                   key.LEFT: False, key.RIGHT: False}

        pgl.gl.glClearColor(1., 1., 1., 1.)

        speed = pgl.text.Label('0 km/h',
                               font_name='Times New Roman',
                               font_size=24,
                               color=(0, 0, 0, 255),
                               x=window.width - 100, y=window.height - 100,
                               anchor_x='center', anchor_y='center')

        # right_bounds = torch.cat((self.right_vecs[:, :-1, :], self.right_vecs[:, 1:, :]), dim=-1)
        # left_bounds = torch.cat((self.left_vecs[:, :-1, :], self.left_vecs[:, 1:, :]), dim=-1)
        # start_bounds = torch.cat((self.left_vecs[:, :1, :], self.right_vecs[:, :1, :]), dim=-1)
        # bounds = torch.cat((right_bounds, left_bounds, start_bounds), dim=1)
        # bounds = bounds.unsqueeze(1).repeat(1, self.num_players, 1, 1).view(-1, *bounds.shape[-2:])
        # reward_line = torch.cat((self.left_vecs[:, -1:, :], self.right_vecs[:, -1:, :]), dim=-1)
        # reward_bound = reward_line.unsqueeze(1).repeat(1, self.num_players, 1, 1).view(-1, *reward_line.shape[-2:])

        bounds = self.bounds
        reward_bound = self.reward_bound

        @window.event
        def on_draw():
            window.clear()

            scale = 200.  # 1unit == 400px
            # center on player
            px, py = scale * self.positions[0, 0, :]
            px = window.width // 2 - px
            py = window.height // 2 - py
            # draw every segment of first track
            batch = pgl.graphics.Batch()
            pgl.gl.glLineWidth(10)  # px
            for x1, y1, x2, y2 in bounds[0, :, :]:
                batch.add(2, pgl.gl.GL_LINES, None,
                          ('v2i', list(map(int, (x1 * scale + px, y1 * scale + py,
                                                 x2 * scale + px, y2 * scale + py)))),
                          ('c3B', (0, 0, 0, 0, 0, 0)))
            fx1, fy1, fx2, fy2 = reward_bound[0, 0, :]
            batch.add(2, pgl.gl.GL_LINES, None,
                      ('v2i', list(map(int, (fx1 * scale + px, fy1 * scale + py,
                                             fx2 * scale + px, fy2 * scale + py)))),
                      ('c3B', (170, 0, 0, 170, 0, 0)))

            pgl.gl.glLineWidth(30)
            dx, dy = scale * 0.1 * self.directions[0, 0, :]
            px, py = window.width // 2, window.height // 2
            batch.add(2, pgl.gl.GL_LINES, None,
                      ('v2i', list(map(int, (px, py, px + dx, py + dy)))),
                      ('c3B', (100, 149, 237, 100, 149, 237)))
            batch.draw()
            speed.text = f'{self.speeds[0, 0] * 36.:7.4f} km/h'
            speed.draw()

        @window.event
        def on_key_press(symbol, modifiers):
            if symbol in actions:
                actions[symbol] = True

        @window.event
        def on_key_release(symbol, modifiers):
            if symbol in actions:
                actions[symbol] = False

        total_time, prev_time = 0., time.time()
        while not self.finished():
            delta = time.time() - prev_time
            prev_time = time.time()
            total_time += delta
            while total_time > 0.:
                total_time -= self.framerate
                fb = (0, 1, 2)[actions[key.UP] - actions[key.DOWN]]
                lr = (0, 3, 6)[actions[key.RIGHT] - actions[key.LEFT]]
                act = torch.zeros((self.num_players, self.num_tracks), dtype=torch.int64, device=self.device)
                act[0, 0] = fb + lr
                self.step(act)

            for wnd in pgl.app.windows:
                wnd.switch_to()
                wnd.dispatch_events()
                wnd.dispatch_event('on_draw')
                wnd.flip()

        window.close()

    @property
    def actions(self):
        return 9

    @staticmethod
    def action_name(a):
        return ('noop', 'forward', 'backward',
                'right', 'forward-right', 'backward-right',
                'left', 'forward-left', 'backward-left')[a]


# ---------------------------------------------------------
# Default params for cars across experiments
# ---------------------------------------------------------


cars = [RaceCar(max_speed=60., acceleration=4., angle=40.),
        RaceCar(max_speed=60., acceleration=1., angle=80.)]


game = Race(timeout=40., framerate=1. / 20., cars=cars)
