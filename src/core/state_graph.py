"""
Modified from funzo project (https://github.com/makokal/funzo/blob/master/funzo
/representation/state_graph.py)
"""
from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import warnings

import networkx as nx
from numpy import sqrt, asarray
from six.moves import filter

__all__ = ['StateGraph']


class StateGraph(nx.DiGraph):
    _node_attrs = ('data', 'cost', 'priority', 'Q', 'V', 'pi', 'type')
    _edge_attrs = ('source', 'target', 'duration', 'reward', 'phi', 'traj')

    def __init__(self, **attr):
        """ Generic state graph suited for MDPs

        The state graph encapsulates a flexible representations for an MDP
        which affords use of task specific constraints as well as temporally
        extended actions (in the sense of hierarchical reinforcement
        learning,options)

        Args:
           state_dim (int): The dimensions of the state space used in the graph.

        """
        super(StateGraph, self).__init__(**attr)

    def gna(self, node_id, attribute):
        """ Get a single attribute of a single node

        Parameters
        ------------
        node_id : int
        attribute : string

        """
        self._check_node_attributes(node_id, attribute)
        return self.node[node_id][attribute]

    def sna(self, node_id, attribute, value):
        """ Set a single attribute of a node

        Parameters
        ------------
        node_id : int
        attribute : string
        value : any

        """
        self._check_node_attributes(node_id, attribute)
        self.nodes[node_id][attribute] = value

    def find_neighbors_data(self, c, distance, metric=None):
        """ Find node neighbors based on distance between `data` attribute

        Parameters
        -----------
        c : array-like, shape = N,
            `data` array to search around
        distance: float
            Maximum range for inclusion the returned neighbors list
        metric : callable, optional (default : None)
            Metric function for deciding 'closeness' wrt to `data` attribute
            If `None`, Euclidean distance will be used

        Returns
        -------
        neighbors : list of int
            List of node ids in the "neighborhood"

        Notes
        ------
        Includes the query node in the result

        """
        m = metric
        if metric is None:
            m = eud

        neighbors = filter(lambda n: m(self[n]['data'], c) <= distance,
                           self.nodes())
        return list(neighbors)

    def find_neighbors_range(self, nid, distance):
        """ Find neighboring nodes within a distance

        Parameters
        -----------
        nid : int
            Node id for the query node
        distance: float
            Maximum range for inclusion the returned neighbors list

        Returns
        -------
        neighbors : list of int
            List of node ids in the "neighborhood"

        Notes
        ------
        Includes the query node in the result

        """
        cn = self[nid]['data']
        return self.find_neighbors_data(cn, distance, None)

    def find_neighbors_k(self, nid, k):
        """ Find k nearest neighbors based on Euclidean distance

        The Euclidean distance is computed based on the `data` attribute

        Parameters
        -----------
        nid : int
            Node id for the query node
        k: int
            Maximum number of nodes to return

        Returns
        -------
        neighbors : list of int
            List of node ids in the "neighborhood"

        """
        serch_set = set(self.nodes()) - {nid}
        cn = self[nid]['data']
        distances = {n: eud(self[n]['data'], cn) for n in serch_set}
        sorted_neighbors = sorted(distances.items(), key=lambda x: x[1])
        k_neighbors = sorted_neighbors[:k]
        return list(n[0] for n in k_neighbors)

    def filter_nodes_by_type(self, ntype):
        """ Filter nodes by node type """
        sns = filter(lambda n: self.gna(n, 'type') == ntype, self.node)
        return list(sns)

    def search_path(self, source, target):
        """ Search for a path from ``source`` to ``target`` using A*"""

        def metric():
            if self.has_edge(source, target):
                return -1 * self.get_edge_data(source, target)['reward']
            return 1000

        path = nx.astar_path(self, source, target, heuristic=metric)
        return path

    def get_signal(self, name):
        """ Retrieve a graph signal from the nodes

        The signals correspond to the node attributes in the graph. For Q
        values, the signal is a list of lists, each of varying lengths since
        the number of edges vary per node.

        Parameters
        -----------
        name : str
            Name of signal to retrieve

        Returns
        -------
        signal : array-like
            1D array for Cost, V, and policy; and a list of lists for Q

        """
        if name not in self._node_attrs:
            raise IndexError('Invalid signal name')
        return [self.nodes(data=True)[n][name] for n in self.nodes(True)]

    def save_graph(self, filename):
        """ Save the graph to file """
        nx.write_gpickle(self, filename)

    def load_graph(self, filename):
        """ Load a graph from file """
        nx.read_gpickle(filename)

    def save_svg(self):
        raise NotImplementedError('Not implemented')

    def _check_node_attributes(self, node_id, attribute):
        assert attribute in self._node_attrs, \
            'Attribute [{}] is invalid | Expected:{}'.format(attribute,
                                                             self._node_attrs)
        assert node_id in self.node, \
            'Node ({}) not in the graph'.format(node_id)

    def _check_edge_attributes(self, source, target, attribute):
        assert attribute in self._edge_attrs, \
            'Attribute [{}] is invalid | Expected:{}' \
                .format(attribute, self._edge_attrs)
        assert self.has_edge(source, target), \
            'Edge [{}-{}] does not exist in the graph'.format(source, target)

    @property
    def transition_matrix(self):
        """ Get the transition matrix T(s, a, s')

        Obtained from the adjacency matrix of the underlying graph

        """
        return nx.adjacency_matrix(self).todense()


def eud(data1, data2):
    return sqrt((data1[0] - data2[0]) ** 2 + (data1[1] - data2[1]) ** 2)
