from Graph import Graph
from Node import Node
from Cluster import Cluster
from Edge import Edge
import numpy as np
from scipy import sparse

limlog = np.vectorize(lambda x, y: np.log(x / y) if x > 0 and y > 0 else 0.)


def steady_state(transition: sparse.spmatrix, damping: float = .85, tol: float = 1e-6):
    N = transition.shape[0]
    antidamp = (1 - damping) / N
    matrix = transition * damping

    stationary = np.ones(N, dtype=np.float32) / N
    next_stationary = stationary @ matrix + antidamp

    while np.linalg.norm(next_stationary - stationary) > tol:
        stationary = next_stationary
        next_stationary = stationary @ matrix + antidamp

    return sparse.csr_matrix(next_stationary)


def invert(matrix):
    sparse_matrix = matrix.copy()
    sparse_matrix.data = 1 / sparse_matrix.data

    return sparse_matrix


def clusters(membership, node=None):
    return membership.getrow(node).nonzero()[1] if node is not None else np.unique(membership.nonzero()[1])


def clique(membership, node=None, cluster=None):
    if node is not None:
        return membership.getcol(clusters(membership, node=node)[0]).nonzero()[0]
    elif cluster is not None:
        return membership.getcol(cluster).nonzero()[0]


def louvain(states, joint):
    init_status = states.copy()
    states = states.tocsc()
    joint = joint.copy().tocsr()

    N = states.shape[1]
    joint_itself = sparse.csr_matrix(joint.diagonal())
    joint.setdiag(0)
    joint_others = joint + joint.T
    membership = sparse.csr_matrix(np.eye(N))

    run = True

    actual_states = states.copy()
    init_membership = membership.copy()

    jointer = joint_others.copy() / 2
    jointer.setdiag(joint_itself.data)
    r = 0
    prev_c = 0
    while run:
        r += 1
        print(r, 'FULLRAN')
        impact_itself = joint_itself.copy()
        impact_others = joint_others.copy()
        run = False
        change = True
        clusters_actual = clusters(membership)
        if r > 1:
            temp_membership = sparse.dok_matrix((N, N))
            temp_init_membership = sparse.dok_matrix((N, N))
            for node in clusters(init_membership):
                temp_init_membership[clique(init_membership, cluster=node), clusters(membership, node=node)] = 1
            for node in clusters_actual:
                temp_membership[node, node] = 1
            init_membership = temp_init_membership
            membership = temp_membership

        print(objective(init_membership, init_status, jointer))
        while change:
            print('RUN', len(clusters(membership)))
            change = False
            c = 0
            s = 0
            for node in clusters_actual:
                member = clusters(membership, node=node)
                node_impact = impact_others.getrow(node)
                node_state = states[:, node][0, 0]
                node_joint = joint_itself[:, node][0, 0]
                connected = node_impact.nonzero()[1]
                connected = np.delete(connected, connected == member)
                if len(connected) > 0:
                    before_member_joint = impact_itself[:, member][0, 0]
                    before_member_state = actual_states[:, member][0, 0]
                    after_member_joint = before_member_joint - node_joint - node_impact[:, member][0, 0]
                    after_member_state = before_member_state - node_state
                    itself_change = before_member_joint * limlog(before_member_joint, before_member_state ** 2) + \
                                    (before_member_state - before_member_joint) * limlog(
                        (before_member_state - before_member_joint),
                        (1 - before_member_state) * before_member_state) - \
                                    after_member_joint * limlog(after_member_joint, after_member_state ** 2) - \
                                    (after_member_state - after_member_joint) * limlog(
                        (after_member_state - after_member_joint), (1 - after_member_state) * after_member_state)
                    before_move_joint = np.squeeze(impact_itself[:, connected].toarray())
                    before_move_states = np.squeeze(actual_states[:, connected].toarray())
                    after_move_joint = before_move_joint + node_joint + np.squeeze(
                        node_impact[:, connected].toarray())
                    after_move_states = before_move_states + node_state
                    others_change = after_move_joint * limlog(after_move_joint, after_move_states ** 2) + \
                                    (after_move_states - after_move_joint) * limlog(
                        (after_move_states - after_move_joint), (1 - after_move_states) * after_move_states) - \
                                    before_move_joint * limlog(before_move_joint, before_move_states ** 2) - \
                                    (before_move_states - before_move_joint) * limlog(
                        (before_move_states - before_move_joint), (1 - before_move_states) * before_move_states)
                    others_change = np.array([others_change]).flatten()
                    max_change = np.argmax(others_change)
                    if others_change[max_change] > itself_change:
                        c += 1
                        s += others_change[max_change] - itself_change
                        if prev_c == 12:
                            print('node', node)
                            print(others_change, max_change)
                            print(after_move_joint * limlog(after_move_joint, after_move_states ** 2),
                                  (after_move_states - after_move_joint) * limlog(
                                      (after_move_states - after_move_joint),
                                      (1 - after_move_states) * after_move_states),
                                  before_move_joint * limlog(before_move_joint, before_move_states ** 2),
                                  (before_move_states - before_move_joint) * limlog(
                                      (before_move_states - before_move_joint),
                                      (1 - before_move_states) * before_move_states))
                            print(others_change[max_change], itself_change)
                            print('from', member)
                            print(before_member_joint,
                                  before_member_state,
                                  after_member_joint,
                                  after_member_state)
                            print('to', connected[max_change])
                            try:
                                print(before_move_joint[max_change],
                                      before_move_states[max_change],
                                      after_move_joint[max_change],
                                      after_move_states[max_change])
                            except IndexError:
                                print(before_move_joint,
                                      before_move_states,
                                      after_move_joint,
                                      after_move_states)
                        run = True
                        change = True
                        member_change = sparse.dok_matrix((N, N))
                        member_change[node, connected[max_change]] = 1
                        member_change[node, member] = -1
                        membership += member_change

                        impact_itself += node_joint * member_change[node, :] + node_impact.multiply(
                            member_change.getrow(node))
                        impact_others += joint_others[:, node] * member_change[node, :]
                        from_state = actual_states[:, member]
                        to_state = actual_states[:, connected[max_change]]
                        from_state.data -= node_state
                        to_state.data += node_state
                        actual_states[:, member] = from_state
                        actual_states[:, connected[max_change]] = to_state
                        impact_others.data = np.maximum(impact_others.data, 0)
                        impact_itself.data = np.maximum(impact_itself.data, 0)
                        actual_states.data = np.maximum(actual_states.data, 0)
            print(c, s)
            prev_c = c
        states = actual_states.copy()
        joint = (membership.T.dot(impact_others / 2) + sparse.diags(
            np.squeeze((joint_itself @ membership).toarray()))).tolil()
        joint_itself = sparse.csr_matrix(joint.diagonal())
        joint.setdiag(0)
        joint_others = joint + joint.T
    return init_membership


def cluster_states(membership, node_states):
    return sparse.csr_matrix(node_states.T.multiply(membership).sum(axis=0))


def cluster_joint(membership, node_joint, diagonal_only=False):
    N = membership.shape[0]
    all_clusters = np.unique(membership.nonzero()[1])
    if diagonal_only:
        joint = sparse.dok_matrix((1, N))
        for i in all_clusters:
            joint[0, i] = membership.getcol(i).multiply(membership.getcol(i).T).multiply(node_joint).sum()
    else:
        joint = sparse.dok_matrix((N, N))
        for i in all_clusters:
            for j in all_clusters:
                joint[i, j] = (membership.getcol(i).multiply(membership.getcol(j).T).multiply(node_joint)).sum()

    return joint


def objective(membership, node_states, node_joint):
    states = cluster_states(membership, node_states)
    nz = states.nonzero()
    states = states[nz].getA()[0]
    joint = cluster_joint(membership, node_joint, diagonal_only=True)[nz].toarray()[0]
    res = sum(joint * limlog(joint, states ** 2) + (states - joint) * limlog(states - joint, states * (1 - states)))
    print('obj', res)

    return res


def joint_matrix(states, adjacency):
    return adjacency.multiply(states.T)


def adjacency_matrix(states, joint):
    return joint.multiply(invert(states).T)


def hamiltonian(node, adjacency, cluster_nodes):
    if len(cluster_nodes) > 1:
        return -adjacency[cluster_nodes, node].sum() - adjacency[node, cluster_nodes].sum()
    return -adjacency[cluster_nodes[0], node] - adjacency[node, cluster_nodes[0]]


def heat_bath(membership, adjacency, steps, temperature, boltzmann=1.):
    membership = membership.copy()
    print('temp', temperature)
    choices = np.random.randint(membership.shape[0], size=steps)
    for node in choices:
        print('GEN', node)
        all_clusters = clusters(membership)
        node_cluster = clusters(membership, node=node)
        print('CLUSTERS')
        hamiltonians = np.vectorize(lambda cluster: hamiltonian(node, adjacency,
                                                                clique(membership, cluster=cluster)))(all_clusters)
        print('HAMS')
        accepts = np.vectorize(lambda hamilton: np.exp(-hamilton / temperature / boltzmann) if hamilton < 0 else 0)(
            hamiltonians)
        probas = np.cumsum(accepts) / sum(accepts)
        destination = np.argmax(np.random.rand() < probas)
        if destination != node_cluster:
            membership[node, node_cluster] = 0
            membership[node, destination] = 1

    return membership


def clustering(adjacency, steps=10, temperatures=np.linspace(1, 5, 5)):
    states = steady_state(adjacency)
    adjacency = adjacency.tolil()
    joint = joint_matrix(states, adjacency)
    membership = louvain(states, joint).tolil()
    adjusted_clusters = np.vectorize(lambda temperature: heat_bath(membership, adjacency, steps, temperature))(
        temperatures)
    results = np.vectorize(lambda cluster_membership: objective(cluster_membership, states, joint))(adjusted_clusters)
    best = np.argmax(results)
    init_result = objective(membership, states, joint)
    best_clustering = adjusted_clusters[best] if results[best] > init_result else membership
    states = cluster_states(best_clustering, states)
    joint = cluster_joint(best_clustering, joint)
    adjacency = adjacency_matrix(states, joint)
    print('FINAL', len(clusters(best_clustering)))

    return best_clustering, states, adjacency
