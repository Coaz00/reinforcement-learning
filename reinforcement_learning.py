import random
import numpy as np
import matplotlib.pyplot as plt

class Environment: 
    
    def __init__(self):
        self.possible_moves = ['U','D','R','L']
        self.active_states = {'A1': 0,'A2': 0,'A3': 0,'A4':0,'A5':0}
        self.terminal_states = {'B1':-1,'B3':-1,'B5':3}
        self.states = {**self.active_states,**self.terminal_states}
        self.state = 'A1' # starting state
        self.is_end = False
        
    def perform_move(self, move):
        if move not in self.possible_moves:
            print("This is an illegal move. Possible moves are U,D,R or L.")
            return False
        else:
            r = random.random()
            if r > 0.6 and r <= 0.8:
                match move:
                    case 'U':
                        move = 'L'
                    case 'L':
                        move = 'D'
                    case 'R':
                        move = 'U'
                    case 'D':
                        move = 'R'
            elif r > 0.8:
                match move:
                    case 'U':
                        move = 'R'
                    case 'L':
                        move = 'U'
                    case 'R':
                        move = 'D'
                    case 'D':
                        move = 'L'
            
            match move:
                case 'U':
                    if self.state[0] == 'B':
                        tmp = self.state.replace('B','A')
                        if tmp in self.states:
                            self.state = tmp
                            
                    return True
                case 'D':
                    if self.state[0] == 'A':
                        tmp = self.state.replace('A', 'B')
                        if tmp in self.states:
                                self.state = tmp
                    
                    return True
                    
                case 'L':
                    tmp = str(int(self.state[1])-1)
                    if (self.state[0] + tmp) in self.states:
                        self.state = self.state[0] + tmp
                        
                    return True
                case 'R':
                    tmp = str(int(self.state[1])+1)
                    if (self.state[0] + tmp) in self.states:
                        self.state = self.state[0] + tmp
            
                    return True
            
                
    def reward(self,state):
        return self.states[state]
    
    def check_end(self):
        if self.state in self.terminal_states:
            self.is_end = True
        else:
            self.is_end = False
            
    def test(self,policy,num_of_episodes):
        
        mean_reward = 0
        mean_turn = 0
        for i in range(num_of_episodes):
            e.reset()
            
            reward = 0
            turn = 0
            while not self.is_end:
                turn+=1
                move = policy[self.state]
                self.perform_move(move)
                reward += self.reward(self.state)
                self.check_end()
        
            mean_reward += reward
            mean_turn += turn
        return mean_reward/num_of_episodes, mean_turn/num_of_episodes
    
    def reset(self):
        self.state = 'A1'
        self.is_end = False
    
    def q_learning(self,gamma,max_episodes):

        self.reset()
        Q = {}
        V = {}
        Vs = []
        Qs = []
        for state in self.states:
            Q[state] = np.zeros(len(self.possible_moves))
            V[state] = 0
        
        epsilon = 1
        rewards = []
        for ep in range(max_episodes):
            
            if (ep+1) % 100 == 0:
                print(ep+1)
                
            alpha = np.log(ep+1)/(ep+1)
            #alpha = 0.01
            
            self.reset()
            reward = 0
            turn = 0
            while not self.is_end:
                
                # choose move
                if random.random() < epsilon:
                    move = random.choice([0,1,2,3])
                else:
                    move = np.argmax(Q[self.state])
                
                current_state = self.state
                
                # play the move
                self.perform_move(self.possible_moves[move])
                reward = reward + gamma**(turn)*self.reward(self.state)
                self.check_end()
                
                # update Q
                Q[current_state][move] = Q[current_state][move] + alpha*(reward + gamma*np.max(Q[self.state])-Q[current_state][move])
                V[current_state] = np.max(Q[current_state])
                
                turn += 1
                

                
            epsilon = np.max([0.01,0.99*epsilon])
            rewards.append(reward)
            Vs.append(V.copy())
            
        return Q, Vs
    
    def score(self,theta,state,move):
        return 1 - np.exp(theta[state][move])/np.sum(np.exp(theta[state]))
        
    
    def play_episode(self,theta):
       
       self.reset()
       states = [self.state]
       rewards = []
       moves = []
       
       while not self.is_end:
           move = np.random.choice([0,1,2,3],p=np.exp(theta[self.state])/np.sum(np.exp(theta[self.state])))
           
           self.perform_move(self.possible_moves[move])
           self.check_end()
           
           rewards.append(self.reward(self.state))
           moves.append(move)
           states.append(self.state)
           
       return states, rewards, moves
        
        
    def reinforce(self, alpha, gamma, max_episodes):
        # init theta
        
        thetaA1u = []
        thetaA1d = []
        thetaA1r = []
        thetaA1l = []
        thetaA2u = []
        thetaA2d = []
        thetaA2r = []
        thetaA2l = []        
        thetaA3u = []
        thetaA3d = []
        thetaA3r = []
        thetaA3l = []
        thetaA4u = []
        thetaA4d = []
        thetaA4r = []
        thetaA4l = []
        thetaA5u = []
        thetaA5d = []
        thetaA5r = []
        thetaA5l = []
        rewards_progress = []
        theta = {}
        for state in self.active_states:
            theta[state] = np.random.rand(4)
        
        for ep in range(max_episodes):
            # play stohastic episode
            states, rewards, moves = self.play_episode(theta)
            
            v = np.zeros(len(rewards))
            for t in range(len(rewards)):
                for tau in range(t,len(rewards)):
                    v[t] += gamma**(tau-t)*rewards[tau]
            
            for i in range(len(moves)):
                state = states[i]
                move = moves[i]
                theta[state][move] += alpha*v[i]*self.score(theta,state,move)
                
            if (ep+1) % 100 == 0:
                thetaA1u.append(theta['A1'][0])
                thetaA1d.append(theta['A1'][1])
                thetaA1r.append(theta['A1'][2])
                thetaA1l.append(theta['A1'][3])
                
                thetaA2u.append(theta['A2'][0])
                thetaA2d.append(theta['A2'][1])
                thetaA2r.append(theta['A2'][2])
                thetaA2l.append(theta['A2'][3])
                
                thetaA3u.append(theta['A3'][0])
                thetaA3d.append(theta['A3'][1])
                thetaA3r.append(theta['A3'][2])
                thetaA3l.append(theta['A3'][3])
                
                thetaA4u.append(theta['A4'][0])
                thetaA4d.append(theta['A4'][1])
                thetaA4r.append(theta['A4'][2])
                thetaA4l.append(theta['A4'][3])
                
                thetaA5u.append(theta['A5'][0])
                thetaA5d.append(theta['A5'][1])
                thetaA5r.append(theta['A5'][2])
                thetaA5l.append(theta['A5'][3])
                
                reward = 0
                for k in range(10):
                    self.reset()
                    while not self.is_end:
                        move = np.random.choice([0,1,2,3],p=np.exp(theta[self.state])/np.sum(np.exp(theta[self.state])))
    
                        self.perform_move(self.possible_moves[move])
                        self.check_end()
                        
                        reward += self.reward(self.state)
                
                rewards_progress.append(reward/10)
                        
        return theta, rewards_progress, thetaA1u, thetaA1d, thetaA1r, thetaA1l, thetaA2u, thetaA2d, thetaA2r, thetaA2l, thetaA3u, thetaA3d, thetaA3r, thetaA3l, thetaA4u, thetaA4d, thetaA4r, thetaA4l, thetaA5u, thetaA5d, thetaA5r, thetaA5l
        
e = Environment()


#%% Q LEARNING
gamma = 0.999
max_episodes = 2500
Q,Vs = e.q_learning(gamma, max_episodes)

policy = {}
for state in e.active_states:
    policy[state] = e.possible_moves[np.argmax(Q[state])]

mean_reward = e.test(policy,10)
print(mean_reward)

def plot(Vs, states):
    for state in states:
        to_plot = []
        for i in range(len(Vs)):
            to_plot.append(Vs[i][state])
        
        plt.figure(1)
        plt.plot(to_plot)
        #plt.show()
    
    plt.figure(1)
    plt.xlabel("# epizoda")
    plt.ylabel("V")
    plt.legend(e.active_states)
    plt.show()
    
plot(Vs,e.active_states)

#%% REINFORCE
theta, rewards_progress, thetaA1u, thetaA1d, thetaA1r, thetaA1l, thetaA2u, thetaA2d, thetaA2r, thetaA2l, thetaA3u, thetaA3d, thetaA3r, thetaA3l, thetaA4u, thetaA4d, thetaA4r, thetaA4l, thetaA5u, thetaA5d, thetaA5r, thetaA5l = e.reinforce(0.4,0.9,15000)


plt.figure()
plt.plot([100*i for i in range(len(rewards_progress))], rewards_progress)
plt.xlabel("# epizoda")
plt.ylabel("Procena prosecne nagrade")
plt.show()

plt.figure()
plt.plot([100*i for i in range(len(rewards_progress))],thetaA1u)
plt.plot([100*i for i in range(len(rewards_progress))],thetaA1d)
plt.plot([100*i for i in range(len(rewards_progress))],thetaA1r)
plt.plot([100*i for i in range(len(rewards_progress))],thetaA1l)
plt.plot([100*i for i in range(len(rewards_progress))],thetaA2u)
plt.plot([100*i for i in range(len(rewards_progress))],thetaA2d)
plt.plot([100*i for i in range(len(rewards_progress))],thetaA2r)
plt.plot([100*i for i in range(len(rewards_progress))],thetaA2l)
plt.plot([100*i for i in range(len(rewards_progress))],thetaA3u)
plt.plot([100*i for i in range(len(rewards_progress))],thetaA3d)
plt.plot([100*i for i in range(len(rewards_progress))],thetaA3r)
plt.plot([100*i for i in range(len(rewards_progress))],thetaA3l)
plt.plot([100*i for i in range(len(rewards_progress))],thetaA4u)
plt.plot([100*i for i in range(len(rewards_progress))],thetaA4d)
plt.plot([100*i for i in range(len(rewards_progress))],thetaA4r)
plt.plot([100*i for i in range(len(rewards_progress))],thetaA4l)
plt.plot([100*i for i in range(len(rewards_progress))],thetaA5u)
plt.plot([100*i for i in range(len(rewards_progress))],thetaA5d)
plt.plot([100*i for i in range(len(rewards_progress))],thetaA5r)
plt.plot([100*i for i in range(len(rewards_progress))],thetaA5l)
plt.xlabel("# epizoda")
plt.ylabel("vrednost parametara")
plt.show()

for state in theta:
    print(np.exp(theta[state])/(np.sum(np.exp(theta[state]))))