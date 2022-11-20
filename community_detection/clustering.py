from Graph import Graph
from Node import Node
from Cluster import Cluster
from Edge import Edge
import numpy as np
from copy import deepcopy
from scipy import sparse

import warnings
warnings.filterwarnings("error")


def partial_objectivity(joint, state):

    try:
        inside = joint * np.log(joint / state ** 2)
    except RuntimeWarning:
        inside = 0

    try:
        outside = (state - joint) * np.log((state - joint) / state / (1 - state))
    except RuntimeWarning:
        outside = 0

    return max(0, inside + outside)


def partial_objectivity_change(current_graph, init_graph, source, moving, target):

    joint = current_graph._cluster_joint
    states = current_graph._cluster_states
    init_joint = init_graph._cluster_joint
    init_states = init_graph._cluster_states

    moving_state = init_states[moving.name]
    moving_joint = init_joint[moving.name, moving.name]
    """
    joint_source_prev = joint[source.name, source.name]
    joint_source_next = joint_source_prev - (
            sum(init_joint[source.nodes, moving.name] + init_joint[moving.name, source.nodes]) - moving_joint)

    state_source_prev = states[source.name]
    state_source_next = state_source_prev - moving_state

    joint_target_prev = joint[target.name, target.name]
    joint_target_next = joint_target_prev + (
        sum(init_joint[target.nodes, moving.name] + init_joint[moving.name, target.nodes]) + moving_joint)

    state_target_prev = states[target.name]
    state_target_next = state_target_prev + moving_state"""

    def p1(n, sign):
        joint_prev = joint[n.name, n.name]
        joint_next = joint_prev + sign * sum(init_joint[n.nodes, moving.name] + init_joint[moving.name, n.nodes]) + moving_joint
        return joint_prev, joint_next

    def p2(n, sign):
        state_prev = states[n.name]
        state_next = state_prev + sign * moving_state
        return state_prev, state_next

    joint_source_prev, joint_source_next = p1(source, -1)
    state_source_prev, state_source_next = p2(source, -1)
    joint_target_prev, joint_target_next = p1(target, 1)
    state_target_prev, state_target_next = p2(target, 1)

    return partial_objectivity(joint_source_next, state_source_next) + partial_objectivity(joint_target_next, state_target_next) -\
            partial_objectivity(joint_source_prev, state_source_prev) - partial_objectivity(joint_target_prev, state_target_prev)


def move_cluster(current_graph, init_graph, source, moving, target):

    joint = current_graph._cluster_joint
    states = current_graph._cluster_states
    init_joint = init_graph._cluster_joint
    init_states = init_graph._cluster_states

    row_move = init_joint[moving.name, :]
    col_move = init_joint[:, moving.name]

    def joint_sum(cluster):

        return sum(row_move[cluster.nodes] + col_move[cluster.nodes]) - init_joint[moving.name, moving.name]

    def update_joint(cluster, sign):

        row_cluster = row_move.copy()
        col_cluster = col_move.copy()
        row_cluster[cluster.nodes] = 0
        col_cluster[cluster.nodes] = 0
        for node in current_graph.neighbours(moving):
            row = row_cluster[node.name]
            row_cluster[node.name] -= row
            row_cluster[node.cluster] += row

            col = col_cluster[node.name]
            col_cluster[node.name] -= col
            col_cluster[node.cluster] += col

        joint[cluster.name, :] += sign * row_cluster
        joint[:, cluster.name] += sign * col_cluster
        joint[cluster.name, cluster.name] += sign * joint_sum(cluster)

    update_joint(source, -1)
    source.remove_node(moving)

    moving.cluster = target.name

    target.append_node(moving)
    update_joint(target, 1)

    states[source.name] -= init_states[moving.name]
    states[target.name] += init_states[moving.name]

    return joint, states


def louvain(graph, tol=1e-6, maxiter=10):

    states = graph._node_states.tolil()
    joint = graph._node_joint.tolil()
    N = graph.no_of_nodes()
    joint_itself = sparse.csr_matrix(joint.diagonal())
    joint.setdiag(0)
    joint_others = joint + joint.T

    impact_itself = joint_itself.copy()
    impact_others = joint_others.copy()
    membership = sparse.eye(N, N)

    # ex
    node = 0
    member = membership.getrow(node).nonzero()[1]
    node_impact = impact_others.getrow(node)
    node_joint = joint_others.getrow(node)
    impact_member = impact_itself[:, node]
    state_member = states[:, node]
    connected = node_impact.nonzero()[1]
    connected = np.delete(connected, connected == member)

    before_member_joint = node_joint[:, member]
    before_member_state = node_joint[:, member]
    after_member_joint = before_member_joint - impact_member - node_impact[:, member]
    after_member_state = before_member_state - state_member
    itself_change = before_member_joint * np.log(before_member_joint / before_member_state ** 2) + \
    (before_member_state - before_member_joint) * np.log(
        (before_member_state - before_member_joint) / (1 - before_member_state) / before_member_state) - \
    after_member_joint * np.log(after_member_joint / after_member_state ** 2) - \
    (after_member_state - after_member_joint) * np.log(
        (after_member_state - after_member_joint) / (1 - after_member_state) / after_member_state)

    before_move_joint = node_joint[:, connected]
    before_move_states = node_joint[:, connected]
    after_move_joint = np.squeeze((before_move_joint + impact_member + node_impact[:, connected]).asarray())
    after_move_states = np.squeeze((before_move_states + state_member).asarray())
    others_change = after_move_joint * np.log(after_move_joint / after_move_states ** 2) + \
                    (after_move_states - after_move_joint) * np.log((after_move_states - after_move_joint) / (1 - after_move_states) / after_move_states) - \
                    before_move_joint * np.log(before_move_joint / before_move_states ** 2) - \
                    (before_move_states - before_move_joint) * np.log((before_move_states - before_move_joint) / (1 - before_move_states) / before_move_states)
    max_change = np.argmax(others_change)
    if others_change[max_change] > itself_change:
        member_change = sparse.dok_matrix(N, N)
        member_change[node, max_change] = 1
        member_change[node, member] = -1
        membership += member_change
        impact_others += impact_others @ member_change
        impact_itself += impact_itself @ member_change + (member_change * impact_others).sum(axis=0)

    global_change_flag = True
    N = graph.no_of_clusters()
    measures = np.zeros((N, N))
    is_measured = np.zeros((N, N), dtype=bool)
    while global_change_flag:
        global_change_flag = False
        temp_graph = Graph([Node(i) for i in range(graph.no_of_clusters())], graph._cluster_edges, graph._cluster_states)
        old = temp_graph.objectivity()

        local_change_flag = True
        i = 0

        def change():

            def change1():
                temp_states = temp_graph._cluster_states.copy().reshape(-1, 1)
                temp_states[temp_states == 0] = 1
                return temp_states

            temp_states = change1()

            def change2():

                temp_graph._cluster_adjacency = temp_graph._cluster_joint / temp_states

            change2()

        while local_change_flag and i < maxiter:
            i += 1
            local_change_flag = False
            for moving in temp_graph._nodes:
                neighbours = temp_graph.neighbours(moving)
                cluster_neighbourhood = list(set([temp_graph.get_cluster(neighbour) for neighbour in neighbours]).difference({temp_graph.get_cluster(moving)}))
                cluster_neighbourhood = list(filter(lambda cluster: not is_measured[moving.cluster, cluster.name], cluster_neighbourhood))
                if len(cluster_neighbourhood) > 0:
                    cluster_names = [cluster.name for cluster in cluster_neighbourhood]
                    measures[moving.name, cluster_names] = np.vectorize(lambda target: partial_objectivity_change(temp_graph, graph, temp_graph.get_cluster(moving), moving, target))(cluster_neighbourhood)
                    is_measured[moving.name, cluster_names] = True
                changes = measures[moving.cluster, :]
                best_change = np.argmax(changes)
                max_change = changes[best_change]
                if max_change > 0:
                    def nulling():
                        cluster = temp_graph.get_cluster(moving)
                        measures[cluster.nodes, :] = 0
                        measures[:, cluster.name] = 0
                        is_measured[cluster.nodes, :] = 0
                        is_measured[:, cluster.name] = 0

                    nulling()
                    temp_graph._cluster_joint, temp_graph._cluster_states = move_cluster(temp_graph, graph,
                                                                                         temp_graph.get_cluster(moving),
                                                                                         moving,
                                                                                         temp_graph._clusters[best_change])
                    nulling()

            change()
            new = temp_graph.objectivity()
            if new - old > tol:
                local_change_flag = True
                global_change_flag = True
                old = new

        def update_cluster():
            new_clusters = []
            to_delete = []
            j = 0
            for i, cluster in enumerate(temp_graph._clusters):
                if len(cluster.nodes) > 0:
                    temp_cluster = []
                    for node_cluster in cluster.nodes:
                        temp_cluster += graph._clusters[node_cluster]._nodes
                        for node in graph._clusters[node_cluster]._nodes:
                            node.cluster = j
                    new_clusters.append(temp_cluster)
                    j += 1
                else:
                    to_delete.append(i)

            return new_clusters, to_delete

        new_clusters, to_delete = update_cluster()

        def update_graph():
            graph._clusters = [Cluster(i, nodes) for i, nodes in enumerate(new_clusters)]
            graph._cluster_joint = np.delete(np.delete(temp_graph._cluster_joint, to_delete, axis=0), to_delete, axis=1)
            graph._cluster_states = np.delete(temp_graph._cluster_states, to_delete)
            graph._cluster_adjacency = graph._cluster_joint / graph._cluster_states.reshape(-1, 1)
            graph._cluster_edges = [Edge(i, j, graph._cluster_adjacency[i, j]) for i in range(graph.no_of_clusters()) for j in range(graph.no_of_clusters()) if graph._cluster_adjacency[i, j] > 0]
        update_graph()

        true_clusters = [cluster.name for cluster in temp_graph._clusters if len(cluster.nodes) > 1]

        def update_measures(arr):

            arr[true_clusters, :] = 0
            arr = np.delete(np.delete(arr, to_delete, axis=0), to_delete, axis=1)
            return arr

        measures = update_measures(measures)
        is_measured = update_measures(is_measured)

    return graph


def move_node(source, moving, target):

    source.remove_node(moving)
    target.append_node(moving)

    return moving


def hamiltonian(node, adjacency, cluster_nodes):

    deltas = np.zeros(adjacency.shape[0])
    deltas[cluster_nodes] = 1
    deltas[node] = 0

    return -sum(adjacency[:, node] + adjacency[node, :] * deltas)


def heat_bath(graph, steps, temperature, boltzmann=1.):

    temp_graph = deepcopy(graph)
    for step in range(steps):
        node = np.random.randint(temp_graph.no_of_nodes())
        hamiltionians = np.vectorize(lambda cluster: hamiltonian(node, temp_graph._node_adjacency, [temp_graph._node_dict[clustered] for clustered in cluster.nodes]))(temp_graph._clusters)
        accepts = np.exp(-hamiltionians / temperature / boltzmann)
        probas = np.cumsum(accepts / sum(accepts))
        move_node(temp_graph.get_cluster(node), node, temp_graph._clusters[np.argmax(np.random.rand() < np.cumsum(probas))])

    temp_graph.recalculate()
    return temp_graph


def clustering(graph, steps=100, temperatures=np.linspace(0, 4, 41)):

    graph = louvain(graph)
    graphs = np.vectorize(lambda temperature: heat_bath(graph, steps, temperature))(temperatures)
    results = np.vectorize(lambda g: g.objectivity())(graphs)
    best_graph = np.argmax(results)
    if results[best_graph] > graph.objectivity():
        return graphs[best_graph]

    return graph


