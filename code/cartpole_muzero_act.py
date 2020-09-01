from cartpole_muzero_train import naive_search
import gym
import tensorflow as tf
import numpy as np



env = gym.make("LunarLander-v2")
env.observation_space, env.action_space
# can act?
# sort of
current_state = env.reset()
mu = tf.keras.models.load_model('../Training/model/mu')
done = False
while not(done):
  policy = naive_search(current_state, model=mu)
  a_1 = np.random.choice([0,1,2,3], p=policy)
  # v,aa = ret[0]
  # print(aa[0], v)
  env.render()
  _,r,done,_ = env.step(a_1)
  if done:
    print("DONE", sn)
    break