import sys
from mpi4py import MPI

from architecture.pulsar import Pulsar


comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()
agent_index = int(sys.argv[1])

actor_pulsar = Pulsar(training=True)
training_pulsar = Pulsar(training=True)
'''
# Inputs to build network
scalar_features = {'match_time': np.array([[120], [110]])}
entities = np.array([[[0, 1, 0], [1, 0, 0]], [[0, 0, 1], [1, 1, 0]]], dtype=np.float32)
entity_masks = np.array([[0, 1], [1, 0]], dtype=np.float32)
pulsar(scalar_features, entities, entity_masks)
'''
weights = comm.recv()
pulsar.set_weights(weights)

while True:
    trajectory = MPI.COMM_WORLD.recv()
    nsteps = len(trajectory['mb_obs'][0])

'''
b = MPI.COMM_WORLD.send("From learner", dest=0)
c = MPI.COMM_WORLD.recv()
print("Received msg:", c)
'''
