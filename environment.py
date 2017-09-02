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
		#print(exchange.id, markets)
		self.action_space = None
		self.observation_space = None

	def _step(self, action):
		"""
		Returns four values:
		- observation (object): an environment-specific object representing your
								observation of the environment. For example, the
								order book for a trading pair.
		- reward (float): amount of reward achieved by the previous action.
		- done (boolean): whether it's time to reset the environment again. When
						  done is True, the episode has terminated (e.g. a
						  specified threshold of money lost has been reached.)
		- info (dict): diagnostic information useful for debugging. It can
					   sometimes be useful for learning (for example, it might
					   contain the raw probabilities behind the environment's
					   last state change). However, official evaluations of your
					   agent are not allowed to use this for learning. (This
					   could be the price history for a trading pair?)
		"""
		pass

	def _reset(self):
		pass

	def _render(self, mode='human', close=False):
		pass

	def _close(self):
		pass

	def _seed(self):
		pass

class MarketSpace(gym.Space):
	"""
	https://gym.openai.com/docs
	"""
	def __init__(self):
		pass