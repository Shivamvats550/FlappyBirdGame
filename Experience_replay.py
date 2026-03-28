from collections import deque
import random
class ReplayMemory():

    # create replay memory/fifo queue
    def __init__(self,maxlen,seed=None):  # seed is basicaly used for randomlysample memory
            self.memory = deque([],maxlen=maxlen)

    def append(self,new_exp):
          self.memory.append(new_exp)

    def sample(self,sample_size):
          return random.sample(self.memory,sample_size)
    
    # curr buffer size
    def __len__(self):
          return len(self.memory)
        
