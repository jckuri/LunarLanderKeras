import collections
import gym
import keras
import numpy
import random
import time

def reshape_state(state):
 return numpy.reshape(state, [1, state.shape[0]])

class LunarLanderDDQN:
 
 experiment_id = 0

 def __init__(self, max_episode = 4000, alpha = 5e-5, gamma = 1.0 - 1e-3, hidden_neurons = 222, hidden_layers = 3):
  LunarLanderDDQN.experiment_id += 1
  self.environment = gym.make('LunarLander-v2')
  self.filename = 'id_{}_alpha_{}_gamma_{}_neurons_{}_layers_{}'.format(LunarLanderDDQN.experiment_id, alpha, gamma, hidden_neurons, hidden_layers)
  self.show_animation = False
  self.n_states = self.environment.observation_space.shape[0]
  self.n_actions = self.environment.action_space.n
  self.alpha = alpha
  self.gamma = gamma
  self.epsilon = 1.0
  self.epsilon_decay = 1e-3
  self.minimum_epsilon = 1e-3
  self.experiences_tuples = collections.deque(maxlen = 40000)
  self.minibatch_size = 32
  self.hidden_1 = hidden_neurons
  self.hidden_2 = hidden_neurons
  self.hidden_3 = hidden_neurons if hidden_layers == 3 else None
  self.max_episode = max_episode
  self.nn_to_train = self.create_neural_network()
  self.static_nn = self.create_neural_network()
  self.update_static_neural_network()

 def create_neural_network(self):
  nn = keras.models.Sequential()
  nn.add(keras.layers.Dense(self.hidden_1, input_dim = self.n_states, activation = 'relu'))
  nn.add(keras.layers.Dense(self.hidden_2, activation = 'relu'))
  if self.hidden_3 is not None:
   nn.add(keras.layers.Dense(self.hidden_3, activation = 'relu'))
  nn.add(keras.layers.Dense(self.n_actions, activation = 'linear'))
  nn.compile(loss = 'mse', optimizer = keras.optimizers.Adam(lr = self.alpha))
  return nn

 def decay_epsilon(self):
  if self.epsilon > self.minimum_epsilon: self.epsilon -= self.epsilon_decay

 def update_static_neural_network(self):
  self.static_nn.set_weights(self.nn_to_train.get_weights())

 def suggest_action(self, state):
  if numpy.random.rand() <= self.epsilon: return random.randrange(self.n_actions)
  return numpy.argmax(self.nn_to_train.predict(state)[0])

 def add_experience_tuple(self, experience_tuple):
  self.experiences_tuples.append(experience_tuple)

 def train_neural_network(self):
  if len(self.experiences_tuples) < self.minibatch_size * 2: return
  minibatch = numpy.array(random.sample(self.experiences_tuples, self.minibatch_size))
  state_1 = numpy.concatenate(minibatch[:, 0])
  action = numpy.array(minibatch[:, 1], dtype = numpy.int)
  reward = minibatch[:, 2]
  state_2 = numpy.concatenate(minibatch[:, 3])
  terminal = minibatch[:, 4]
  actions_1 = self.nn_to_train.predict(state_1)
  actions_2 = self.nn_to_train.predict(state_2)
  static_actions_2 = self.static_nn.predict(state_2)
  for i in range(self.minibatch_size):
   actions_1[i, action[i]] = reward[i] + (0.0 if terminal[i] else self.gamma * static_actions_2[i, numpy.argmax(actions_2[i])])
  #actions_1[:, action] = reward + numpy.where(terminal, numpy.zeros((self.minibatch_size, )), self.gamma * static_actions_2[:, numpy.argmax(actions_2, axis = 1)])
  self.nn_to_train.fit(state_1, actions_1, batch_size = self.minibatch_size, epochs = 1, verbose = 0)
  
 def execute_once_in_a_while(self, frame):
  self.train_neural_network()
  if frame % 400 == 0: self.decay_epsilon()
  if frame % 5000 == 0: self.update_static_neural_network()

 def win(self):
  ddqn_file = 'data/{}.h5'.format(self.filename)
  self.nn_to_train.save_weights(ddqn_file)
  print('THE DDQN WAS SUCCESSFULLY TRAINED AND SAVED TO DISK: {}, moving_average_score: {}'.format(ddqn_file, self.moving_average_score))
  self.log_file.write('SUCCESS!')

 def run_episodes(self):
  log_filename = 'data/{}.log'.format(self.filename)
  print('Writing log: {}'.format(log_filename))
  self.log_file = open(log_filename, "w")
  self.log_file.write("Episode,Score,Moving Average Score,Epsilon,Seconds\n")
  last_scores = collections.deque(maxlen = 100)
  episode = 0
  frame = 0
  time0 = time.time()
  while True:
   episode += 1
   state_1 = reshape_state(self.environment.reset())
   score = 0.0
   terminal = False
   while not terminal:
    frame += 1
    if self.show_animation: self.environment.render()
    action = self.suggest_action(state_1)
    state_2, reward, terminal, _ = self.environment.step(action)
    state_2 = reshape_state(state_2)
    score += reward
    self.add_experience_tuple((state_1, action, reward, state_2, terminal))
    state_1 = state_2
    self.execute_once_in_a_while(frame)
    if episode >= self.max_episode: break
   last_scores.append(score)
   self.moving_average_score = numpy.mean(last_scores)
   dt = time.time() - time0
   print("episode: {}, score: {}, moving_average_score: {}, epsilon: {:.4}, seconds: {}".format(episode, score, self.moving_average_score, self.epsilon, dt))
   self.log_file.write('{},{},{},{:.4},{}\n'.format(episode, score, self.moving_average_score, self.epsilon, dt))
   if self.moving_average_score >= 200:
    self.win()
    break
   if episode >= self.max_episode: break
  self.log_file.close()

def main():
 max_ep = 3200
 for alpha in [5e-7, 5e-5, 5e-3, 5e-1]:
  LunarLanderDDQN(max_episode = max_ep, alpha = alpha).run_episodes()
 for gamma in [0.9999, 0.999, 0.99, 0.9]:
  LunarLanderDDQN(max_episode = max_ep, gamma = gamma).run_episodes()
 for hidden_layers in [3, 2]:
  for inc in [-100, 0, 100]:
   LunarLanderDDQN(max_episode = max_ep, hidden_neurons = 222 + inc, hidden_layers = hidden_layers).run_episodes()

if __name__ == "__main__": main()
