from __future__ import division, absolute_import

import json
from collections import Callable

from sklearn import gaussian_process
import numpy as np

from .state_graph import StateGraph


class ControllerGraph(object):
    """ A Controller Graph

    A graph based representations of continuous state MDPs by sampling a few
    states and connecting them using local controllers (which can be seen as
    Markov options). Generates a very sparse representations which is efficient
    and also allows for task constraints to be modeled directly into the MDP
    action space via these local controllers.

    Requires partially specified MDP with reward function, and a way to check
    set of terminal states. The transition function role is handles by the
    local controller.

    """

    def __init__(self, params, starts, goal, controller, state_dim=3):
        self._params = params
        self._controller = controller
        self._starts = starts
        self._goal = goal

        # setup the graph structure and internal variables
        self._g = StateGraph(state_dim=state_dim)
        self._best_trajs = []
        self._node_id = 0
        self._max_conc = 1.0
        self._max_es = 1.0
        self._min_es = 0.0

        self._gp = gaussian_process.GaussianProcess(corr='squared_exponential',
                                                    theta0=1e-2,
                                                    thetaL=1e-4,
                                                    thetaU=1e-1)

    def initialize_state_graph(self, R, terminal, samples=None):
        """ Initialize the controller graph

        Parameters
        -----------
        R : callable
            Reward function callable to evaluating reward along local
            controller trajectories
        terminal : callable
            Function to check is a state is terminal.
        samples : array-like, optional (default: None)
            Samples of states for initializing the graph

        """
        if not isinstance(R, Callable):
            raise TypeError('*R* must be a callable')
        if not isinstance(terminal, Callable):
            raise TypeError('*terminal* must be a callable')


        self._g.clear()

        self._fixed_init(R, terminal, samples)


    def build_graph(self):
        """ Build the controller graph """
        pass

    def states(self):
        """ Return the ids of the states in the CG """
        return self._g.nodes

    def all_actions(self):
        """ Return the ids of the edges in the CG """
        return self._g.edges

    def state_actions(self, state):
        """ Get the actions available at state (out-going) edges """
        return self._g.out_edges(state)

    def _fixed_init(self, R, terminal, samples):
        """ Initialize from random samples """

        # CMAX = self._params.max_cost
        CMAX = 100
        RMAX = 1

        for start in self._starts:
            self._g.add_node(nid=self._node_id, data=start, cost=0,
                             priority=1, V=RMAX, pi=0, Q=[], ntype='start')
            self._node_id += 1

        self._g.add_node(nid=self._node_id, data=self._goal, cost=-CMAX,
                         priority=1, V=RMAX, pi=0, Q=[], ntype='goal')
        self._node_id += 1

        # add the initial samples of states
        init_samples = list(samples)
        for sample in init_samples:
            self._g.add_node(nid=self._node_id, data=sample, cost=-CMAX,
                             priority=1, V=RMAX, pi=0, Q=[], ntype='simple')
            self._node_id += 1

        # add edges between each pair of states
        for n in self._g.nodes:
            for m in self._g.nodes:
                if n == m or terminal(n):
                    continue
                ndata, mdata = self._g.gna(n, 'data'), self._g.gna(m, 'data')
                traj = self._controller.trajectory(ndata, mdata)
                d = trajectory_length(traj)
                r, phi = 1, [0, 1, 1]
                self._g.add_edge(source=n, target=m, reward=r,
                                 duration=d, phi=phi, traj=traj)

    def _traj_init(self, R, terminal, trajectories):
        """ Initialize CG using way-point samples from expert trajectories """
        pass


class CGParameters(object):
    """ ControllerGraph parameters

    Uses JSON encoding for persistence, so as to allow different domain
    configurations.

    Attributes
    ------------
    n_expand : int
        Number of nodes to expand at every iteration
    n_new : int
        Number of new nodes to sample around the expansion node (represents
        the number of trials)
    n_add : int
        Number of nodes to add to the graph from the set of sampled nodes.
        Allows sampling and ranking to select only top-K nodes
    radius : float
        Radius around a node to sample new nodes from (effectively controls
        the extent of running the local controller)
    exp_thresh : float
        Threshold on the value of a node to select it for expansion
    max_traj_len : int
        Maximum allowed length of trajectories (policies) in the CG
    p_best : float
        Probability of expanding from the set of nodes which are part of the
        current best policies to the goal at every iteration
    max_samples : int
        Maximum number of samples/nodes for the CG. Used for termination in the
        building of the graph
    max_edges : int
        Maximum number of edges allowed. Controls sparsity of the graph and
        also helps to encourage exploration
    init_type : string
        CG initialization type, either: random or trajectory
            * random -- sample initial set of nodes uniformly from the world
            * trajectory -- sample the initial set of nodes from provided
            expert demonstration trajectories.
    max_cost : float
        TODO
    conc_scale : None
        TODO
    tmin : float
        Lowe, time limit for running the local controller. Tempered with
        iterations.
    tmax : float
        Upper time limit for running the local controller. Tempered with
        iterations.

    """

    _PARAMS = [
        'n_expand',
        'n_new',
        'n_add',
        'radius',
        'exp_thresh',
        'max_traj_len',
        'p_best',
        'max_samples',
        'max_edges',
        'init_type',
        'max_cost',
        'conc_scale',
        'tmin',
        'tmax',
    ]

    def __init__(self, **kwargs):
        self.n_expand = kwargs.pop('n_expand', 1)
        self.n_new = kwargs.pop('n_new', 20)
        self.n_add = kwargs.pop('n_add', 1)
        self.radius = kwargs.pop('radius', 1.8)
        self.exp_thresh = kwargs.pop('exp_thresh', 1.2)
        self.max_traj_len = kwargs.pop('max_traj_len', 500)
        self.p_best = kwargs.pop('p_best', 0.4)
        self.max_samples = kwargs.pop('max_samples', 100)
        self.max_edges = kwargs.pop('max_edges', 360)
        self.init_type = kwargs.pop('init_type', 'random')
        self.max_cost = kwargs.pop('max_cost', 1000)
        self.conc_scale = kwargs.pop('conc_scale', 1)
        self.tmin = kwargs.pop('tmin', (0.45, 2.4))
        self.tmax = kwargs.pop('tmax', (3.6, 7.2))

    def load(self, json_file):
        """ Load parameters from a json file """
        with open(json_file, 'r') as f:
            jdata = json.load(f)
            for k, v in jdata.items():
                self.__dict__[k] = v

    def save(self, filename):
        """ Save the parameters to file """
        with open(filename, 'w') as f:
            json.dump(self._to_dict(), f, indent=4, sort_keys=True)

    def __repr__(self):
        return self._to_dict()

    def __str__(self):
        d = self._to_dict()
        return ''.join('{}: {}\n'.format(k, v) for k, v in d.items())

    def _to_dict(self):
        return self.__dict__


def trajectory_length(traj):
    """ Compute the length of a 2D trajectory

    Parameters
    -----------
    traj : array-like
        Trajectory representing the path traveled as an array of
        shape [n_waypoints x frame_size]. Frame encodes information at
        every time step e.g [x, y, theta, ...]

    Returns
    ---------
    path_length : float
        Length of the path as a sum of Euclidean distance between successive
        way-points.
    """

    traj = np.asarray(traj)
    if traj.ndim != 2:
        raise ValueError("Trajectory must be a two dimensional array")

    d = np.diff(traj[:, 0:2], axis=0)
    n = np.linalg.norm(d, axis=1)

    return np.sum(n)