import numpy as np
import gym

#Make the environment & get to the start state
env = gym.make("MountainCar-v0")
initState=env.reset()

numSteps = 100
qtableSize = (numSteps,numSteps,env.action_space.n)

#Initialize q-table
qtable = np.random.uniform(-2,0,qtableSize)

#Determine the low and high values of state space for discretization
low=env.observation_space.low
high=env.observation_space.high
step=(high-low)/numSteps

#Go from actualState to the discrete qtable index
def getDiscreteState(state):
    return tuple(((state-low)//step).astype(int))

eps=0.0      #Exploration rate
lr=0.5        #learning_rate
gamma=0.9     #discount factor

numEpisodes=10000
for episode in range(numEpisodes):
    eps = 0.1/np.sqrt(1+episode)
    done=False
    best=-2
    currState=env.reset()
  
    while(not done):
        currStateId=getDiscreteState(currState)

        if(random.random()<eps):
            action=env.action_space.sample()
        else:
            action=np.argmax(qtable[currStateId])

        nextState, reward, done, info = env.step(action)
        nextStateId=getDiscreteState(nextState)
        best=max(best,nextState[0])
    
        if(done):
            break
        else:
            #Q-learning Update
            qtable[currStateId+(action,)]+=lr*(reward+gamma*np.max(qtable[nextStateId])-qtable[currStateId+(action,)])

        currState=nextState
    if(best>0.5):
        print(episode,": Mission Accomplished!!!",best)
    else:
        print(episode,": Training in progress",best)

env.close()