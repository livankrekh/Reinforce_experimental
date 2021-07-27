#!./venv/bin/python3.7

import gym
import torch
import numpy as np

from torch import nn
from torch.distributions.normal import Normal

import torch.nn.functional as funcs
from tqdm import tqdm

env = gym.make('Marvin-v0')
env._max_episode_steps = 5000
env.reset()

RANDOM_STATE = 123

ZERO_LIMIT = 10**-9

class PolicyNetwork(nn.Module):

	def __init__(self, input_size=24, n_actions=4, random_state=None):
		super(PolicyNetwork, self).__init__()
		self.fc1 = nn.Linear(input_size, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, n_actions)

		self.input_size = input_size
		self.n_actions = n_actions

		if random_state != None:
			torch.manual_seed(random_state)
			np.random.seed(random_state)

	def forward(self, x):
		if not isinstance(x, torch.Tensor):
			x = torch.tensor(x, requires_grad=True).float()

		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		mean = torch.tanh(self.fc3(x))
		std = torch.exp(self.fc3(x))

		return mean, std

	def train(self, actions, states, rewards, new_states, batch_size=1024, epochs=5, optim=torch.optim.Adam, lr=10**-4):
		all_actions = torch.tensor(actions, requires_grad=True).float()
		all_states = torch.tensor(states, requires_grad=True).float()
		all_rewards = torch.tensor(rewards, requires_grad=True).float()
		all_new_states = torch.tensor(new_states, requires_grad=True).float()
		optimizer = optim(self.parameters())

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
		X = torch.tensor(X, requires_grad=False).float()
		mean, _ = self.forward(X)

		return mean

	def loss_function(self, mean, std, actions, rewards, states, new_states):
		dist = Normal(mean, std)
		probs = dist.cdf(actions)
		total_rewards = self._rewards_policy(rewards, states, new_states)

		loss = -torch.sum(torch.log(probs) * total_rewards.expand_as(probs))

		return loss

	def _rewards_policy(self, rewards, states, new_states, delta=1., coff=10):
		total_rewards = []
		total = .0
		size = len(rewards)

		for i in reversed(range(len(rewards))):
			curr_delta = delta ** (size - i)
			total += rewards[i] * curr_delta / (size - i)

			total_rewards.insert(0,total)

		tens = torch.tensor([total_rewards], requires_grad=True).reshape((-1,1))
		velocity = torch.absolute(states[:,1]).reshape((-1,1)) * coff
		pos = torch.absolute(new_states[:,0] - states[:,0]).reshape((-1,1)) * coff
		acc = (torch.absolute(new_states[:,1]).reshape((-1,1)) - torch.absolute(states[:,1]).reshape((-1,1))) * coff

		return tens.reshape((-1,1)) + velocity

def simulation(network, n):

	for _ in range(n):
		done = False
		observation = env.reset()

		while not done:
			env.render()
			observation = np.array(observation).reshape((-1,2))

			action = network.predict(observation).detach().numpy().reshape((4,))
			observation, reward, done, info = env.step(action)

			print(action)

if __name__ == "__main__":
	state_size = 24
	actions_size = 4
	n_training_episodes = 10
	n_test_episodes = 10

	policy_nn = PolicyNetwork(input_size=state_size, n_actions=actions_size, random_state=RANDOM_STATE)

	for i_episode in tqdm(range(n_training_episodes)):
		observation = env.reset()
		done = False

		actions = []
		observations = []
		rewards = []
		new_observations = []

		while not done:
			env.render()
			observation = np.array(observation)
			random_action = np.random.uniform(low=-1., high=1., size=(1,))

			observations.append(observation)
			actions.append(random_action)

			observation, reward, done, info = env.step(random_action)

			new_observations.append(observation)
			rewards.append(reward)

		policy_nn.train(np.array(actions), np.array(observations), np.array(rewards), np.array(new_observations), epochs=2)

	simulation(policy_nn, n_test_episodes)
