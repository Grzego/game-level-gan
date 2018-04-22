import math
import torch

from utils.pytorch_utils import cudify, tensor_from_list
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
        self.cars_max_speed = tensor_from_list([car.max_speed for car in cars])
        self.cars_acceleration = tensor_from_list([car.acceleration for car in cars])
        self.cars_angle = tensor_from_list([car.angle for car in cars])
        self.positions = None
        self.directions = None
        self.speeds = None
        self.alive = None
        self.steps = 0
        self.steps_limit = int(timeout // framerate)
        self.bounds = None
        self.reward_bound = None
        self.action_speed = tensor_from_list([0.,   # noop
                                              1.,   # forward
                                              -1.,  # backward
                                              0.,   # left
                                              0.,   # right
                                              1.,   # forward-left
                                              1.,   # forward-right
                                              -1.,  # backward-left
                                              -1.,  # backward-right
                                              ])
        self.action_dirs = tensor_from_list([0.,   # noop
                                             0.,   # forward
                                             0.,   # backward
                                             1.,   # left
                                             -1.,  # right
                                             1.,   # forward-left
                                             -1.,  # forward-right
                                             1.,   # backward-left
                                             -1.,  # backward-right
                                             ])
        self.observation_size = observation_size
        self.max_distance = max_distance

    def state_shape(self):
        return self.observation_size,

    def players_layer_shape(self):
        pass

    def reset(self, tracks):
        """
        tracks - [num_boards, segments, (arc, width)]

        where:
            arc is in range (-1, 1) meaning next segment angle from (-90deg, 90deg)
            width is in range (0, 1) meaning width from (0.5, 1.)
        """
        self.steps = 0

        # add sentinels to tracks (0 in front and 0 in back of track for additional segment)
        num_boards = tracks.size(0)
        tracks = torch.cat((cudify(torch.zeros(num_boards, 2, 2)), tracks, cudify(torch.zeros(num_boards, 1, 2))),
                           dim=1)

        arcsum = 0.5 * math.pi * torch.cumsum(tracks[:, :, :1], dim=1)  # cumsum over angles and conversion to radians
        segment_vecs = torch.cat((torch.cos(arcsum), torch.sin(arcsum)), dim=2)

        signs = torch.sign(tracks[:, 1:, :1])
        signs[signs == 0.] = 1.

        right_vecs = -segment_vecs[:, :-1, :] + segment_vecs[:, 1:, :]
        right_vecs = right_vecs * signs / torch.sqrt(torch.sum(torch.pow(right_vecs, 2.), dim=-1, keepdim=True))
        right_vecs *= 0.5 + 0.5 * tracks[:, 1:, 1:]
        left_vecs = -right_vecs

        segments = torch.cumsum(segment_vecs[:, 1:, :], dim=1)
        right_vecs = segments + right_vecs
        left_vecs = segments + left_vecs

        right_bounds = torch.cat((right_vecs[:, :-1, :], right_vecs[:, 1:, :]), dim=-1)
        left_bounds = torch.cat((left_vecs[:, :-1, :], left_vecs[:, 1:, :]), dim=-1)
        start_bounds = torch.cat((left_vecs[:, :1, :], right_vecs[:, :1, :]), dim=-1)
        self.reward_bound = torch.cat((left_vecs[:, -1:, :], right_vecs[:, -1:, :]), dim=-1)
        self.bounds = torch.cat((right_bounds, left_bounds, start_bounds), dim=1)

        self.positions = cudify(torch.zeros(num_boards, self.num_players, 2))
        self.positions[:, :, 1] = 0.1  # 1m after a start
        self.directions = cudify(torch.zeros(num_boards, self.num_players, 2))
        self.directions[:, :, 1] = 1.  # direction vectors should have length = 1.
        self.speeds = cudify(torch.zeros(num_boards, self.num_players))
        self.alive = cudify(torch.ByteTensor(num_boards, self.num_players))
        self.alive.fill_(True)

    @staticmethod
    def _segment_collisions(segments, tests):
        """
        Tests collision between every S and P for every N.
        Returns True/False wherther collision occured (shape [N, S * P])

        segments = [N, S, 4]
        tests = [N, P, 4]

        N - usually batch size or number of boards
        S - number of segments
        P - number of players
        """
        n = segments.size(0)

        def _orientation(p, q, r):
            """
            Returns orientation of r point with respect to segment p -> q
            (-1, 0, 1) - (widdershins - counterclockwise, clockwise, colinear)
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
            return (q <= torch.max(p, r)) & (q >= torch.min(p, r))

        p1, q1 = segments[:, :, :2], segments[:, :, 2:]
        p2, q2 = tests[:, :, :2], tests[:, :, 2:]

        o1 = _orientation(p1, q1, p2).view(n, -1)  # [N, S * P]
        o2 = _orientation(p1, q1, q2).view(n, -1)  # [N, S * P]
        o3 = _orientation(p2, q2, p1).permute(0, 2, 1).view(n, -1)  # [N, S * P] - permute to match cases
        o4 = _orientation(p2, q2, q1).permute(0, 2, 1).view(n, -1)  # [N, S * P]

        # general case
        res = (o1 != o2) & (o3 != o4)

        # special cases
        res |= (o1 == 0.) & _on_segment(p1, q1, p2).view(n, -1)
        res |= (o2 == 0.) & _on_segment(p1, q1, q2).view(n, -1)
        res |= (o3 == 0.) & _on_segment(p2, q2, p1).permute(0, 2, 1).view(n, -1)
        res |= (o4 == 0.) & _on_segment(p2, q2, q1).permute(0, 2, 1).view(n, -1)

        return res

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
        far_dirs[:, :, 2:] *= 10000.
        collisions = Race._segment_collisions(segments, far_dirs).view(n, s, d)  # [N, S, D]

        p, q = segments[:, :, None, :2], segments[:, :, None, 2:]
        s, d = directions[:, None, :, :2], directions[:, None, :, 2:]

        qpc = q - p  # [N, S, 1, 2]
        spc = s - p  # [N, S, D, 2]

        dists = spc[:, :, :, 1] * qpc[:, :, :, 0] - spc[:, :, :, 0] * qpc[:, :, :, 1]  # [N, S, D]
        denom = d[:, :, :, 1] * qpc[:, :, :, 0] - d[:, :, :, 0] * qpc[:, :, :, 1]  # [N, S, D]

        dists[collisions] /= denom[collisions]
        dists[~collisions] = float('inf')
        dists[dists < 0.] = float('inf')

        return dists

    @staticmethod
    def _rotate_vecs(vectors, angles):
        """
        Rotates vectors to the left by given angle.

        vectors = [N, 2]
        angles = [N]
        """
        angles = angles.repeat(4).view(-1, 2, 2)
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
        num_boards = actions.size(0)

        #  choose move vector (thats car and action dependent)
        dir_flag = self.action_dirs[actions].view(-1)  # [num_boards * num_players]
        angles = self.framerate * dir_flag * self.cars_angle.repeat(num_boards).view(-1)  # [num_boards * num_players]
        new_dirs = self._rotate_vecs(self.directions.view(-1, 2), angles).view(-1, self.num_players, 2)

        speed_flag = self.action_speed[actions]  # [num_boards, num_players]
        speed = self.speeds + self.framerate * speed_flag * self.cars_acceleration
        speed = torch.min(self.cars_max_speed, torch.max(self.cars_max_speed.neg(), speed))
        new_speed = speed.view(-1, self.num_players, 1)

        new_pos = self.positions + new_dirs * new_speed

        #  check for collisions
        paths = torch.cat((self.positions, new_pos), dim=-1).view(-1, 4)  # [num_boards * num_players, 4]
        # pick positions of alive players
        alive_mask = self.alive.nonzero().view(-1)
        # pick boards with alive players
        bounds_mask = self.alive.max(dim=-1)[0].nonzero().view(-1)

        seg_col = self._segment_collisions(self.bounds[bounds_mask], paths[alive_mask])
        is_dead = seg_col.max(dim=-1)[0]  # shape = [N]
        self.alive[alive_mask] = is_dead

        # update all player variables
        self.directions = new_dirs
        self.speeds = new_speed
        self.positions = new_pos

        #  check for reward
        rewards = cudify(torch.zeros(num_boards, self.num_players))
        rewards[alive_mask] = -1. * is_dead  # -1 reward for each player that died
        is_done = torch.max(self._segment_collisions(self.reward_bound[bounds_mask], paths[alive_mask]))[0]
        rewards[alive_mask] = is_done == True  # cross the finish line; +1 reward
        rewards[rewards == 0.] = -0.01  # small negative reward to encourage finishing race faster

        # return states (state is just distances in few directions)
        obs_angles = torch.linspace(-math.pi / 2., math.pi / 2., self.observation_size)
        obs_angles = obs_angles.repeat(num_boards * self.num_players)
        obs_dirs = self._rotate_vecs(new_dirs.view(-1, 2).repeat(self.observation_size, 1), obs_angles)
        obs_dirs = obs_dirs.view(-1, self.observation_size, 2)  # [num_boards * num_players, observation_size, 2]
        obs_segm = new_pos.view(-1, 1, 2).repeat(1, self.observation_size, 1)
        obs_segm = torch.cat((obs_segm, obs_dirs), dim=-1)

        states = cudify(torch.zeros(num_boards * self.num_players, self.observation_size))
        alive_states = self._smallest_distance(self.bounds[bounds_mask], obs_segm[alive_mask])
        states[alive_mask] = alive_states.clamp(max=self.max_distance)

        return states.permute(1, 0, 2), rewards.t()

    def finished(self):
        #      all players are crushed     or timeout was reached
        return torch.sum(self.alive) < 0.1 or self.steps > self.steps_limit

    @property
    def actions(self):
        return 9

    @staticmethod
    def action_name(a):
        return ('noop', 'forward', 'backward', 'left', 'right',
                'forward-left', 'forward-right', 'backward-left', 'backward-right')[a]
