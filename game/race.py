import math
import torch

from utils.pytorch_utils import device
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
    def __init__(self, timeout, cars: [RaceCar], observation_size=9, max_distance=2., framerate=1. / 30.):
        # 1unit = 10m
        # TODO: try continuous actions
        self.timeout = timeout
        self.framerate = framerate
        self.num_players = len(cars)
        self.cars_max_speed = torch.tensor([car.max_speed for car in cars], dtype=torch.float32, device=device)
        self.cars_acceleration = torch.tensor([car.acceleration for car in cars], dtype=torch.float32, device=device)
        self.cars_angle = torch.tensor([car.angle for car in cars], dtype=torch.float32, device=device)
        self.num_tracks = None
        self.positions = None
        self.directions = None
        self.speeds = None
        self.alive = None
        self.scores = None
        self.steps = 0
        self.steps_limit = int(timeout // framerate)
        self.bounds = None
        self.reward_bound = None
        self.action_speed = torch.tensor([0.,  # noop
                                          1.,  # forward
                                          -1.,  # backward
                                          0.,  # right
                                          1.,  # forward-right
                                          -1.,  # backward-right
                                          0.,  # left
                                          1.,  # forward-left
                                          -1.,  # backward-left
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
        self.history = []  # keeps posision and direction for players on first board

    def state_shape(self):
        return self.observation_size,

    def players_layer_shape(self):
        pass

    def reset(self, tracks):
        """
        tracks - [num_boards, segments, (arc, width)]

        where:
            arc is in range (-1, 1) meaning next segment angle from (-90deg, 90deg)
            width is in range (0, 1) meaning width from (0.15, 0.5)

        # TODO: test different settings
        """
        self.steps = 0
        self.num_tracks = tracks.size(0)
        self.history = []

        # add sentinels to tracks (0 in front and 0 in back of track for additional segment)
        num_boards = tracks.size(0)
        tracks = torch.cat((torch.zeros((num_boards, 1, 2), device=device), tracks,
                            torch.zeros((num_boards, 1, 2), device=device)),
                           dim=1)

        arcsum = 0.3 * math.pi * torch.cumsum(tracks[:, :, :1], dim=1)  # cumsum over angles and conversion to radians
        segment_vecs = torch.cat((torch.sin(arcsum), torch.cos(arcsum)), dim=2)

        perp_vecs = segment_vecs.clone()
        perp_vecs[:, :, 0], perp_vecs[:, :, 1] = segment_vecs[:, :, 1], -segment_vecs[:, :, 0]

        right_vecs = perp_vecs[:, 1:, :] + perp_vecs[:, :-1, :]
        right_vecs /= right_vecs.norm(p=2., dim=-1, keepdim=True)
        right_vecs *= 0.15 + 0.35 * tracks[:, :-1, 1:]
        right_vecs = torch.cat((torch.zeros((num_boards, 1, 2), device=device), right_vecs), dim=1)
        right_vecs[:, 0, 0] = 0.15
        left_vecs = -right_vecs

        segments = torch.cumsum(segment_vecs, dim=1)
        segments[:, 1:, :] = segments[:, :-1, :].clone()
        segments[:, 0, :] = 0.
        right_vecs = segments + right_vecs
        left_vecs = segments + left_vecs

        right_bounds = torch.cat((right_vecs[:, :-1, :], right_vecs[:, 1:, :]), dim=-1)
        left_bounds = torch.cat((left_vecs[:, :-1, :], left_vecs[:, 1:, :]), dim=-1)
        start_bounds = torch.cat((left_vecs[:, :1, :], right_vecs[:, :1, :]), dim=-1)
        reward_bound = torch.cat((left_vecs[:, -1:, :], right_vecs[:, -1:, :]), dim=-1)
        reward_bound = reward_bound.repeat(1, self.num_players, 1, 1)
        self.reward_bound = reward_bound.view(-1, *reward_bound.shape[-2:])
        bounds = torch.cat((right_bounds, left_bounds, start_bounds), dim=1).unsqueeze(1)
        bounds = bounds.repeat(1, self.num_players, 1, 1)
        self.bounds = bounds.view(-1, *bounds.shape[-2:])  # [num_boards * num_players, num_segments, 4]

        self.positions = torch.zeros((num_boards, self.num_players, 2), device=device)
        self.positions[:, :, 1] = 0.1  # 1m after a start
        self.directions = torch.zeros((num_boards, self.num_players, 2), device=device)
        self.directions[:, :, 1] = 1.  # direction vectors should have length = 1.
        self.speeds = torch.zeros((num_boards, self.num_players), device=device)
        self.alive = torch.ones((num_boards, self.num_players), dtype=torch.uint8, device=device)
        self.scores = torch.empty((num_boards, self.num_players), dtype=torch.float32, device=device)
        self.scores.fill_(float('inf'))  # time when finished

        return self.step(torch.zeros((num_boards, self.num_players), dtype=torch.int64, device=device))[0]

    @staticmethod
    def _segment_collisions(segments, tests, special=False):
        """
        Tests collision between every S and P for every N.
        Returns True/False wherther collision occured (shape [N, S, P])

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

        p1, q1 = segments[:, :, :2], segments[:, :, 2:]
        p2, q2 = tests[:, :, :2], tests[:, :, 2:]

        o1 = _orientation(p1, q1, p2).view(n, -1)  # [N, S]
        o2 = _orientation(p1, q1, q2).view(n, -1)  # [N, S]
        o3 = _orientation(p2, q2, p1).permute(0, 2, 1).contiguous().view(n, -1)  # [N, S] - permute to match cases
        o4 = _orientation(p2, q2, q1).permute(0, 2, 1).contiguous().view(n, -1)  # [N, S]

        # general case
        res = (o1 != o2) & (o3 != o4)

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

    def step(self, actions):
        """
        actions - torch.Tensor with size [num_boards, num_players]
        """
        self.steps += 1
        num_boards, num_seg = actions.size(0), self.bounds.size(2)

        # if player won/died ignore it's action
        actions[~self.alive] = 0  # noop

        #  choose move vector (thats car and action dependent)
        dir_flag = self.action_dirs[actions.view(-1)]  # [num_boards * num_players]
        angles = self.framerate * dir_flag * self.cars_angle.repeat(num_boards).view(-1)  # [num_boards * num_players]
        new_dirs = self._rotate_vecs(self.directions.view(-1, 2), angles).view(-1, self.num_players, 2)

        speed_flag = self.action_speed[actions.view(-1)].view(num_boards, -1)  # [num_boards, num_players]
        speed = self.speeds + self.framerate * speed_flag * self.cars_acceleration[None, :]
        new_speed = torch.min(self.cars_max_speed, torch.max(self.cars_max_speed.neg(), speed))
        is_moving = torch.abs(new_speed.view(-1)) > 1e-7  # [num_boards * num_players]

        new_pos = self.positions + new_dirs * new_speed[:, :, None]

        # pick boards with alive players
        update_mask = (self.alive.view(-1) & is_moving).nonzero().squeeze(-1)

        rewards = torch.empty((num_boards * self.num_players), device=device)
        rewards.fill_(-0.01)  # small negative reward over time

        if update_mask.numel() > 0:  # anyone moved
            # check collisions
            paths = torch.cat((self.positions, new_pos), dim=-1).view(-1, 1, 4)  # [num_boards * num_players, 1, 4]

            seg_col = self._segment_collisions(self.bounds[update_mask], paths[update_mask])
            is_dead = seg_col.squeeze(dim=-1).max(dim=1)[0]
            self.alive.view(-1)[update_mask] = ~is_dead

            #  check for reward
            # +1 if finished
            # -1 if died
            # -0.01 otherwise to encourage finishing race faster
            finish = self._segment_collisions(self.reward_bound[update_mask], paths[update_mask])
            is_done = torch.max(finish.view(update_mask.size(0), -1), dim=-1)[0]
            rewards[update_mask] += is_done.float() - is_dead.float()

            self.alive.view(-1)[update_mask] &= ~is_done
            # TODO: if wins/dies -> store scores
            # stop finished players
            new_speed[~self.alive] = 0.

        # update all player variables
        self.directions = new_dirs
        self.speeds = new_speed
        self.positions = new_pos

        # return states (state is just distances in few directions)
        obs_angles = torch.linspace(-math.pi / 2., math.pi / 2., self.observation_size).to(device).view(1, -1)
        obs_angles = obs_angles.repeat(num_boards * self.num_players, 1).view(-1)
        rot_dirs = new_dirs.view(-1, 1, 2).repeat(1, self.observation_size, 1).view(-1, 2)
        obs_dirs = self._rotate_vecs(rot_dirs, obs_angles)
        obs_dirs = obs_dirs.view(-1, self.observation_size, 2)  # [num_boards * num_players, observation_size, 2]
        obs_segm = new_pos.view(-1, 1, 2).repeat(1, self.observation_size, 1)
        obs_segm = torch.cat((obs_segm, obs_dirs), dim=-1)

        states = torch.zeros((num_boards * self.num_players, self.observation_size), device=device)
        alive_mask = self.alive.view(-1).nonzero().squeeze(-1)
        if alive_mask.numel() > 0:
            alive_states = self._smallest_distance(self.bounds[alive_mask], obs_segm[alive_mask])
            states[alive_mask] = alive_states.clamp(max=self.max_distance)

        # record history
        self.history.append((self.positions[0].tolist(), self.directions[0].tolist()))

        return states.view(num_boards, self.num_players, -1).permute(1, 0, 2), rewards.view(num_boards, -1).t()

    def finished(self):
        #      all players are crushed     or timeout was reached
        return torch.sum(self.alive).item() < 0.1 or self.steps > self.steps_limit

    def record_episode(self, filename: str):
        """
        Saves a movie with gameplay on first track to a file.
        Requires OpenCV and movie-py.

        filename - without extension

        This method is rather inefficient and should be rarely used.
        """
        import os
        import cv2
        import moviepy.editor as mpy
        import numpy as np

        width, height = 320, 240
        scale = 100.  # 1unit == 400px

        record = 255 * np.ones((self.num_players, len(self.history), height, width, 3), dtype=np.uint8)

        for frame, (positions, directions) in enumerate(self.history):
            for player, (position, direction) in enumerate(zip(positions, directions)):
                # center on player
                px, py = position
                px *= scale
                py *= scale
                px = width // 2 - px
                py = height // 2 - py
                # draw every segment of first track
                for x1, y1, x2, y2 in self.bounds[0, :, :]:
                    x1, y1, x2, y2 = map(int, (x1 * scale + px, y1 * scale + py,
                                               x2 * scale + px, y2 * scale + py))
                    cv2.line(record[player, frame], (x1, height - y1), (x2, height - y2), (0, 0, 0, 0),
                             thickness=3, lineType=cv2.LINE_AA)

                fx1, fy1, fx2, fy2 = self.reward_bound[0, 0, :]
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

        record = np.concatenate(list(record), axis=-2)

        basepath, _ = os.path.split(filename)
        os.makedirs(basepath, exist_ok=True)
        clip = mpy.ImageSequenceClip(list(record), fps=30)
        clip.write_videofile(filename + '.mp4', audio=False, verbose=False)

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
            for x1, y1, x2, y2 in self.bounds[0, :, :]:
                batch.add(2, pgl.gl.GL_LINES, None,
                          ('v2i', list(map(int, (x1 * scale + px, y1 * scale + py,
                                                 x2 * scale + px, y2 * scale + py)))),
                          ('c3B', (0, 0, 0, 0, 0, 0)))
            fx1, fy1, fx2, fy2 = self.reward_bound[0, 0, :]
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
                act = torch.zeros((self.num_tracks, self.num_players), dtype=torch.int64, device=device)
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
