import gym

#Make the environment & get to the start state
env = gym.make("MountainCar-v0")
initState=env.reset()

#Understanding the state,action space of the environment
print("Action space is:",list(range(env.action_space.n)))
print("State space is between",env.observation_space.low,'&',env.observation_space.high)
#Here,the state is an array of position and velocity of the car

done = False
numIters=20
iter=0

while(not done and iter<numIters):
    #env.render()
    action=env.action_space.sample()
    nextState,reward, done, info = env.step(action) # take a random action
    #print(nextState,reward)
    iter+=1

env.close()

