# Source https://github.com/geohot/ai-notebooks. Thanks to George
# This implemmetation is for Lunar Lander

# %pylab inline
import tensorflow as tf
#import tensorflow.keras.backend as K
import numpy as np
import gym
from tqdm import tqdm,trange
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from scipy.special import softmax
# np.set_printoptions(suppress=True)

# Create the Lunar Lander Env

env = gym.make("LunarLander-v2")
env.observation_space, env.action_space

# Here we create the 3 models that makes MuZero
S_DIM = 4

# h: representation function
# s_0 = h(o_1...o_t)
x = o_0 = Input(env.observation_space.shape)
x = Dense(64)(x)
x = Activation('elu')(x)
s_0 = Activation('tanh')(x)
s_0 = Dense(S_DIM, name='s_0')(x)
h = Model(o_0, s_0, name="h")
def ht(o_0):
  return h.predict(np.array(o_0)[None])[0]

# g: dynamics function (recurrent in state?) old_state+action -> state+reward
# r_k, s_k = g(s_k-1, a_k)
s_km1 = Input(S_DIM)
a_k = Input(env.action_space.n)
x = Concatenate()([s_km1, a_k])
x = Dense(64)(x)
x = Activation('elu')(x)
x = Dense(64)(x)
x = Activation('elu')(x)
s_k = Dense(S_DIM, name='s_k')(x)
r_k = Dense(1, name='r_k')(x)
g = Model([s_km1, a_k], [r_k, s_k], name="g")
g.compile('adam', 'mse')
def gt(s_km1, a_k):
  r_k, s_k = g.predict([s_km1[None], a_k[None]])
  return r_k[0], s_k[0]

# f: prediction function -- state -> policy+value
# p_k, v_k = f(s_k)
x = s_k = Input(S_DIM)
x = Dense(32)(x)
x = Activation('tanh')(x)
p_k = Dense(env.action_space.n)(x)
p_k = Activation('softmax', name='p_k')(p_k)
v_k = Dense(1, name='v_k')(x)
f = Model(s_k, [p_k, v_k], name="f")
f.compile('adam', 'mse')
def ft(s_k):
  p_k, v_k = f.predict(s_k[None])
  return p_k[0], v_k[0]


# Create the Muzero Function
# It uses dynamic function for rollout search
# K is the number of rollout steps.
# TODO : What is rollout?


K = 5
gamma = 0.95

# represent
o_0 = Input(env.observation_space.shape, name="o_0")
# don't use the h function for now
s_km1 = h(o_0)
# s_km1 = o_0

# rollout with dynamics
# p_k, v_k, r_k = mu(o_0, a_1_k)
a_all, mu_all = [], []
for k in range(K):
  a_k = Input(env.action_space.n, name="a_%d" % k)
  r_k, s_k  = g([s_km1, a_k])

  # predict
  p_k, v_k = f([s_k])

  # store
  a_all.append(a_k)
  mu_all.append([p_k, v_k, r_k])
  s_km1 = s_k

# put in the first observation and actions
#   need policy from search
#   need values from sum of rewards + last state value (real state?)
#   need rewards
#a_all = Concatenate()(a_all)
mu = Model([o_0, a_all], mu_all)
mu.compile('adam', 'mse')
mu.summary()


# toh -> to_one_hot
def to_one_hot(x,n):
  ret = np.zeros([n])
  ret[x] = 1.0
  return ret
# Enumerate the whole action space.
import itertools

# aopts = list(itertools.product([0,1], repeat=K))
# aoptss = np.array([[toh(x, 2) for x in aa] for aa in aopts])
aopts = list(itertools.product([0,1,2,3], repeat=K))
aoptss = np.array([[to_one_hot(x, 4) for x in aa] for aa in aopts])
aoptss = aoptss.swapaxes(0,1)
aoptss = [aoptss[x] for x in range(5)]
# every possible action for the next 5 time 

def naive_search(o_0, model=mu):
  # concatenate the current state with every possible action
  o_0s = np.repeat(np.array(o_0)[None], len(aopts), axis=0)
  ret = mu.predict([o_0s]+aoptss)
  # TODO Check if its -2 or not?
  v_s = ret[-2]
  
  # group the value with the action rollout that caused it
  v = [(v_s[i][0], aopts[i]) for i in range(len(v_s))]
  
  av = [0,0,0,0]
  for vk, ak in v:

    av[ak[0]] += vk
    
  # policy = np.exp(av)/sum(np.exp(av))
  policy = softmax(av)
  return policy
  
  #return sorted(v, reverse=True)

def bstack(bb):
  ret = [[x] for x in bb[0]]
  for i in range(1, len(bb)):
    for j in range(len(bb[i])):
      ret[j].append(bb[i][j])
  return [np.array(x) for x in ret]

# Train the mu model by running many episodes until termination

def run_episode():
  current_state = env.reset()
  # Dat is a sort of replay buffer although George doesn't call it that yet
  dat =[]
  while 1:
    s_0 = np.copy(current_state)
    p_0 = naive_search(s_0)
    # Pick the action based on the probability distribution of the policy given by the naive search.
    # TODO : Make it proportional to the state visit count
    a_1 = np.random.choice([0,1,2,3], p=p_0)
    new_state, r_1, done, _ = env.step(a_1)
    # As per the muzero paper 
    dat.append((s_0, p_0, a_1,r_1))
    current_state = new_state
    # TODO: Handle for env termination because of cap on the time step
    if done:
      break 
  return dat

def get_training_episodes():
  # global replay_buffer
  if random.randint(0,10) == 0 or len(replay_buffer) <10:
    replay_buffer = replay_buffer[0:20]
    dat = run_episode()
    replay_buffer.append(dat)
  else:
    rdat= random.choice(replay_buffer)
  dat = run_episode()

  #TODO: What is this for?

  dat=[]
  for s_0, p_0, a_1, r_1 in rdat:
    dat.append((s_0, naive_search(s_0, a_1, r_1))

  # Compute value function for the complete episode

  v =[0]
  for _, _, _, r_1 in dat[::-1]:

    # TODO: Verify the formula from Sutton and Barto
    v.append(v[-1]*gamma + r_1)
  v = v[::-1][0:-1]


  X, Y =[], []
  for i in range(len(dat)-K):
    x = [dat[i][0]]
    y = []
    for j in range(K):
      x += [to_one_hot(dat[i+j][2], 4)]
      y += [dat[i+j][1], v[i+j], dat[i+j][3]]
    # y += [dat[i+K][1], v[i+k]]
    X.append(x)
    Y.append(y)
  return X,Y



for i in range(100):
  X, Y = [], []
  for _ in range(16):
    x, y = get_training_episodes()
    X +=x
    Y += y
  l1 = mu.fit(bstack(X), bstack(Y), batch_size=16, verbose=0)
  print(l1.history['loss'])

mu.save('../Training/model/mu')
# current_state = env.reset()
# sc = 0
# scs = []
# vs = []
# rs = []
# # "epochs"
# for _ in range(20):
#   X,Y = [],[]
#   for _ in range(16):
#     # TODO: Why does first element is the state and rest actions?P
#     x = [np.copy(current_state)]

#     # actually act with best value policy
#     # TODO: Shouldn't I only take one step?GH
#     y = []
#     for i in range(K):
#       _, v_0 = ft(ht(current_state))
#       p_0 = naive_search(current_state)
#       a_1 = np.random.choice([0,1,2,3], p=p_0)
#       new_state, r_1, done, _ = env.step(a_1)
#       sc += 1
      
#       y += [p_0, None, r_1]
      
#       # append the real actions taken
#       x.append(to_one_hot(a_1, 4))
#       current_state = new_state
    
#     _, v_k = ft(ht(current_state))
#     p_k = naive_search(current_state)
#     y += [p_k, v_k]
    
#     # fix values
#     for i in range(K):
#       y[-4 - i*3] = y[-3 - i*3] + gamma * y[-1 - i*3]
      
#     vs += y[1::3][0:5]
#     rs += y[2::3]
        
#     X.append(x)
#     Y.append(y)
#     if done:
#       env.reset()
#       scs.append(sc)
#       sc = 0
      
#   ll = mu.fit(bstack(X), bstack(Y), verbose=1)
#   loss = ll.history['loss']
#   print(loss)
# plot(vs)
# plot(rs)


