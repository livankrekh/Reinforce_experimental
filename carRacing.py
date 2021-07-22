import gym
import torch
import numpy as np

from torch import nn
from torch.distributions.normal import Normal

import torch.nn.functional as funcs
from tqdm import tqdm

env = gym.make('CarRacing-v0')
# env = gym.make('MountainCarContinuous-v0')

env._max_episode_steps = 1000
gym.logger.set_level(40)
env.reset()

RANDOM_STATE = 123

ZERO_LIMIT = 10**-9

class PolicyNetwork(nn.Module):

	def __init__(self, input_size=24, n_actions=4, random_state=None):
		super(PolicyNetwork, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 3)
		self.conv2 = nn.Conv2d(6, 16, 3)
		self.conv3 = nn.Conv2d(16, 32, 3)
		self.fc1 = nn.Linear(32*2*2, 512)
		self.fc2 = nn.Linear(512, 128)
		self.fc3 = nn.Linear(128, n_actions)

		self.input_size = input_size
		self.n_actions = n_actions

		if random_state != None:
			torch.manual_seed(random_state)
			np.random.seed(random_state)

	def forward(self, x):
		if not isinstance(x, torch.Tensor):
			x = torch.tensor(x, requires_grad=True).float()

		x = torch.max_pool2d(torch.relu(self.conv1(x)), kernel_size=3)
		x = torch.max_pool2d(torch.relu(self.conv2(x)), kernel_size=3)
		x = torch.max_pool2d(torch.relu(self.conv3(x)), kernel_size=3)
		x = x.reshape((-1, 32*2*2))
		x = torch.relu(self.fc1(x))
		x = torch.tanh(self.fc2(x))
		mean = torch.tanh(self.fc3(x))
		std = torch.exp(self.fc3(x))

		return mean, std

	def train(self, actions, states, rewards, new_states, batch_size=25, epochs=5, optim=torch.optim.Adam, lr=10**-6):
		all_actions = torch.tensor(actions).float()
		all_states = torch.tensor(states).float()
		all_rewards = torch.tensor(rewards).float()
		all_new_states = torch.tensor(new_states).float()
		optimizer = optim(self.parameters(), lr=lr)

		losses = []

		actions_batches = torch.split(all_actions, split_size_or_sections=batch_size)
		states_batches = torch.split(all_states, split_size_or_sections=batch_size)
		rewards_batches = torch.split(all_rewards, split_size_or_sections=batch_size)
		new_states_batches = torch.split(all_new_states, split_size_or_sections=batch_size)

		for epoch in range(epochs):
			for action, state, reward, new_state in zip(actions_batches, states_batches, rewards_batches, new_states_batches):

				mean, std = self.forward(state)

				mean.retain_grad()
				std.retain_grad()

				loss = self.loss_function(mean, std, action, reward, state, new_state)

				loss.retain_grad()

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				losses.append(loss.detach().numpy())

			average_loss = np.mean(losses)

	def predict(self, X):
		X = torch.tensor(X).float()
		mean, _ = self.forward(X)

		return mean

	def loss_function(self, mean, std, actions, rewards, states, new_states):
		dist = Normal(mean, std)
		probs = dist.cdf(actions)
		total_rewards = self._rewards_policy(rewards, states, new_states)

		loss = -torch.sum(torch.log(probs) * total_rewards.expand_as(probs))

		return loss

	def _rewards_policy(self, rewards, states, new_states, delta=1., coff=1000):
		total_rewards = []
		total = .0
		size = len(rewards)

		for i in reversed(range(len(rewards))):
			curr_delta = delta ** (size - i)
			total += rewards[i] * curr_delta / (size - i)

			total_rewards.insert(0,total)

		tens = torch.tensor([total_rewards], requires_grad=True).reshape((-1,1))

		return tens.reshape((-1,1))

def simulation(network, n):

	for _ in range(n):
		done = False
		observation = env.reset()

		while not done:
			env.render()
			observation = np.array(observation).reshape((-1,24))

			action = network.predict(observation).detach().numpy().reshape((4,))
			observation, reward, done, info = env.step(action)

if __name__ == "__main__":
	state_size = 24
	actions_size = 3
	n_training_episodes = 10
	n_test_episodes = 10

	policy_nn = PolicyNetwork(input_size=state_size, n_actions=actions_size, random_state=RANDOM_STATE)

	# for i_episode in tqdm(range(n_training_episodes)):
	for i in range(n_training_episodes):
		observation = env.reset()
		done = False

		actions = []
		observations = []
		rewards = []
		new_observations = []

		while not done:
			env.render()
			observation = np.array(observation).reshape((-1,3,96,96))
			if i < 10:
				random_action = np.random.uniform(low=-1., high=1., size=(actions_size,))
			else:
				random_action = policy_nn.predict(observation).detach().numpy().reshape((actions_size,))

			observations.append(observation)
			actions.append(random_action)

			observation, reward, done, info = env.step(random_action)

			new_observations.append(observation)
			rewards.append(reward)

		np_observations = np.array(observations).reshape((-1,3,96,96))
		np_actions = np.array(actions)
		np_rewards = np.array(rewards)
		np_new_observations = np.array(new_observations)

		policy_nn.train(np_actions, np_observations, np_rewards, np_new_observations, epochs=2)

	simulation(policy_nn, n_test_episodes)
