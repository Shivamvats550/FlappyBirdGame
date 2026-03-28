import gymnasium as gym
import flappy_bird_gymnasium
from  DQN_Arcitecture import DQN
import torch 
from Experience_replay import ReplayMemory
import itertools
import torch.nn as nn
import torch.optim as optim
import yaml
import random
import os
import argparse

if torch.backends.mps.is_available():
    device ="mps"
elif torch.cuda.is_available():
    device ="cuda"
else:
    device ="cpu"  #(for in google cloab ) => runtime => change runtime => t4 gpu

RUNS_DIR ="runs"
os.makedirs(RUNS_DIR,exist_ok=True)


class Agent:
    def __init__(self,param_set):
        self.param_set = param_set
        

        with open ("parameters.yaml","r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.alpha =params["alpha"]
        self.gamma =params["gamma"]

        self.epsilon_init =params["epsilon_init"]
        self.epsilon_min =params["epsilon_min"]
        self.epsilon_decay =params["epsilon_decay"]

        self.replay_memory_size=params["replay_memory_size"]
        self.mini_batch_size =params["mini_batch_size"]

        self.reward_threshold =params["reward_threshold"]
        self.netwark_sync_rate =params["netwark_sync_rate"]
        self.mini_batch_size =params["mini_batch_size"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE =os.path.join(RUNS_DIR,f"{self.param_set}.log")
        self.MODEL_FILE =os.path.join(RUNS_DIR,f"{self.param_set}.pt")

    def run(self,is_training =True,render=False):  # is_training => agent is learn for used it ,render=>har bar hmm human training procss in dakh ga  and testing time true 

    
        
       
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None , use_lidar=True)

        num_states =env.observation_space.shape[0] # input dimenstion
        num_actions =env.action_space.n # output dimention
    
        policy_dqn = DQN(num_states,num_actions).to(device)

        
        epsilon = self.epsilon_init

        steps = 0
        if is_training:
            memory = ReplayMemory(self.replay_memory_size)

            # create target netwark
            target_dqn =DQN(num_states ,num_actions).to(device)
            # copy that wt and bias vals form policy => target 
            target_dqn.load_state_dict(policy_dqn.state_dict())

            self.optimizer= optim.Adam(policy_dqn.parameters(),lr=self.alpha)

            best_reward =float("-inf")

        else:
             # bett policy load
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state,dtype =torch.float,device=device)

            episode_reward = 0
            terminated=False

            while (not terminated and episode_reward < self.reward_threshold):
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample() # explore 
                    action = torch.tensor(action,dtype =torch.long,device=device)
                else:
                    with torch.no_grad():
                        
                        action =policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()  # exploit
                # Processing:
                next_state, reward, terminated, _, _ = env.step(action.item())
                episode_reward += reward
                
                # create are tensor
                reward =torch.tensor(reward,dtype=torch.float,device=device)
                next_state =torch.tensor(next_state,dtype=torch.float,device=device)

                
                if is_training :
                    memory.append((state,action,next_state ,reward,terminated))
                    steps+=1

                state = next_state
                episode_reward += reward

            print(f" episode ={episode+1} with total_reward={episode_reward} & epsilon={epsilon}")

            

            if is_training:
                # epsilon_decay=>only for training time 
                epsilon = max(epsilon * self.epsilon_decay,self.epsilon_min)

                if episode_reward > best_reward:
                    log_msg=f"best reward ={episode_reward}for episode={episode+1}"
                    
                    with open(self.LOG_FILE,"a") as f:
                        f.write(log_msg + "\n")

                    torch.save(policy_dqn.state_dict(),self.MODEL_FILE)
                    best_reward =episode_reward
            if is_training and len(memory) > self.mini_batch_size:
                # get sample
                mini_batch =memory.sample(self.mini_batch_size)
                self.optimize(mini_batch,policy_dqn,target_dqn)

            # sync the netwark
            if steps > self.netwark_sync_rate:
                target_dqn.load_state_dict(policy_dqn.state_dict())
                # steps =0

        # env.close() -manually stop

    def optimize(self,mini_batch,policy_dqn,target_dqn):
        # get experience => batch train
        states,actions,next_states ,rewards,terminations =zip(*mini_batch)

        states=torch.stack(states)
        actions =torch.stack(actions)
        next_states =torch.stack(next_states)
        rewards =torch.stack(rewards)
        terminations =torch.tensor(terminations).float().to(device)

           
            # calculate targetyQ-value -if termination =true=>zero
        with torch.no_grad():
            target_q =rewards + (1-terminations) * self.gamma * target_dqn(next_states).max(dim=1)[0]  # max => next state ke liye best action ka q value
            # calculate  y_pred ie.. q-value from current policy 
        current_q =policy_dqn(states).gather(dim=1,index=actions.unsqueeze(dim=1)).squeeze()  # gather => action ke index pr jo q value h use le aao


        # loss 
        loss=self.loss_fn(current_q,target_q)
        # optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
if __name__ == "__main__":
    # parse command line input
    parser = argparse.ArgumentParser(description="Train or test model.")
    parser.add_argument("hyperparameters",help="")
    parser.add_argument("--train",help="Training mode",action="store_true")
    args = parser.parse_args()

    dql = Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False,render=True)

