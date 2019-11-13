import sys
from mpi4py import MPI


comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()
#common_comm = comm.Merge(True)

agent_index = int(sys.argv[1])

while True:
    c = MPI.COMM_WORLD.recv()
    print("Received msg:", c)

'''
b = MPI.COMM_WORLD.send("From learner", dest=0)
c = MPI.COMM_WORLD.recv()
print("Received msg:", c)
'''
