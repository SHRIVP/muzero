from cartpol_muzero_train import naive_search
import gym
import tensorflow as tf



env = gym.make("LunarLander-v2")
env.observation_space, env.action_space
# can act?
# sort of
current_state = env.reset()
mu = tf.keras.models.load_model('../Training/model/mu')
for sn in range(100):
  ret = naive_search(current_state, model=mu)
  v,aa = ret[0]
  print(aa[0], v)
  env.render()
  _,r,done,_ = env.step(aa[0])
  if done:
    print("DONE", sn)
    break