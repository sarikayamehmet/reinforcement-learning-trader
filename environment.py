import gym
from gym import error, spaces, utils
from gym.utils import seeding

import ccxt

class Market(gym.Env):
	"""
	How to subclass Env:
	https://github.com/openai/gym/blob/master/gym/core.py
	"""

	metadata = {'render.modes': ['human']}

	def __init__(self, exchange, symbol):
		self.exchange = exchange
		markets = exchange.load_markets()
		print(exchange.id, markets)

	def _step(self, action):
		pass

	def _reset(self):
		pass

	def _render(self, mode='human', close=False):
		pass

	def _close(self):
		pass

	def _seed(self):
		pass