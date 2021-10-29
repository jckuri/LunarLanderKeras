import lunar_lander_ddqn
import collections
import time
import numpy

class ExpertLunarLander(lunar_lander_ddqn.LunarLanderDDQN):

 def __init__(self, expert_filename = 'expertise'):
  super(ExpertLunarLander, self).__init__()
  self.expert_filename = expert_filename
  nn_filename = '{}.h5'.format(self.expert_filename)
  self.nn_to_train.load_weights(nn_filename)
  self.epsilon = 0.0

 def trials(self):
  log_filename = '{}.log'.format(self.expert_filename)
  self.log_file = open(log_filename, 'w')
  print('Writing log: {}'.format(log_filename))
  self.log_file.write("Episode,Score,Moving Average Score,Epsilon,Seconds\n")
  last_scores = collections.deque(maxlen = 100)
  time0 = time.time()
  for episode in range(1, 100 + 1):
   state_1 = lunar_lander_ddqn.reshape_state(self.environment.reset())
   score = 0.0
   terminal = False
   while not terminal:
    self.environment.render()
    action = self.suggest_action(state_1)
    state_2, reward, terminal, _ = self.environment.step(action)
    state_2 = lunar_lander_ddqn.reshape_state(state_2)
    score += reward
    state_1 = state_2
   last_scores.append(score)
   self.moving_average_score = numpy.mean(last_scores)
   dt = time.time() - time0
   print("episode: {}, score: {}, moving_average_score: {}, epsilon: {:.4}, seconds: {}".format(episode, score, self.moving_average_score, self.epsilon, dt))
   self.log_file.write('{},{},{},{:.4},{}\n'.format(episode, score, self.moving_average_score, self.epsilon, dt))
  self.log_file.close()

if __name__ == "__main__": ExpertLunarLander().trials()
