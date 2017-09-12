import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Box

import numpy as np
import ccxt

#import matplotlib.pyplot as plt

class OrderSpace(gym.Space):
	"""
	Action space for the Market environment.

	- place order (binary): place order or do nothing
	- order type (binary): 'market' or 'limit'
	- side (binary): 'buy' or 'sell'
	- amount (float): how much to trade (could be in base or quote, check API)
	- price (float): percentage of current market price (for limit orders only)
	"""

	side_sign = {'buy': -1, 'sell': 1}

	def __init__(self, max_amount):
		self.max_amount = max_amount

	def sample(self):
		"""
		Uniformly randomly sample a random element of this space
		"""
		self.place_order = np.random.choice([True, False])
		self.order_type = np.random.choice(['market', 'limit'])
		self.side = np.random.choice(['buy', 'sell'])
		self.amount = np.random.uniform(low=0, high=self.max_amount)
		self.price = self.side_sign[self.side]*(1-np.random.random_sample())

		return [self.place_order, self.order_type, self.side, self.amount, self.price]

	def contains(self, x):
		"""
		Return boolean specifying if x is a valid
		member of this space
		"""
		raise NotImplementedError

	def to_jsonable(self, sample_n):
		"""Convert a batch of samples from this space to a JSONable data type."""
		raise NotImplementedError

	def from_jsonable(self, sample_n):
		"""Convert a JSONable data type to a batch of samples from this space."""
		raise NotImplementedError

class Order(object):
	"""
	An object encapsulating an order.
	"""
	def __init__(self, exchange, symbol, place_order, order_type, side, amount, price):
		# Set the attributes of the order.
		self.exchange = exchange
		self.symbol = symbol
		self.place_order = place_order
		self.order_type = order_type
		self.side = side
		self.amount = amount
		self.price = price
		# Initialize the order ID to None.
		self.id = None

	def place(self):
		# If the place_order boolean is true, place an order on the exchange.
		if place_order:
			# If it's a market order, create an order without specifying a price.
			if order_type == 'market':
				order_info = self.exchange.create_order(self.symbol, self.order_type, self.amount)
			# Otherwise, include the price in the order.
			else:
				order_info = self.exchange.create_order(self.symbol, self.order_type, self.amount, self.price)
			# Save the order ID returned from calling create_order.
			self.id = order_info['id']
		# If place_order is false, return None for the order ID.
		else:
			self.id = None
		# Return the order ID.
		return self.id

	def cancel(self):
		self.exchange.cancel_order(self.id)

	def __str__(self):
		return self.id

	def __repr__(self):
		return 


class Market(gym.Env):
	"""
	Market subclasses the OpenAI Gym Env object. It encapsulates a market
	environment, where the action space includes placing and cancelling orders,
	and the observation space includes the order book retrieved at some sampling
	rate. It is a partially observable environment.
	"""

	metadata = {
		'render.modes': ['human', 'ansi']
	}

	# Set the reward range.
	reward_range = (-np.inf, np.inf)

	def __init__(self, exchange, symbol):
		# Load the cryptocurrency exchange.
		self.exchange = exchange
		self.markets = exchange.load_markets()
		self.symbol = symbol

		# Set the max amount (in BTC) per trade.
		self.max_amount = 1.0

		# Set the action space. This is defined by the OrderSpace object.
		self.action_space = OrderSpace(max_amount=self.max_amount)
		
		# Set the observation space. This is the order book. It includes the following:
		# An arbitrarily long number of columns, where each column has:
		# - A discrete variable {-1, 1} indicating a bid or an ask.
		# - A continuous variable [0, inf) for the price.
		# - A continuous variable [0, inf) for the quantity.
		self.observation_space = Box(np.array([0,0.0,0.0]), np.array([np.inf, np.inf, np.inf]))

		# Set the seed for the environment's random number generator.
		self.seed()

		# Reset the environment.
		self.reset()

	def _observe(self):
		# Fetch the order book (dictionary) for our symbol.
		order_book = self.exchange.fetch_order_book(self.symbol)

		# Calculate the market price.
		bid = order_book['bids'][0][0] if len (order_book['bids']) > 0 else None
		ask = order_book['asks'][0][0] if len (order_book['asks']) > 0 else None
		spread = (ask - bid) if (bid and ask) else None

		# Put the bids and asks into separate arrays.
		bids = np.array(order_book['bids'])
		asks = np.array(order_book['asks'])

		# Label the bids with -1 and the asks with 1.
		bid_sign = -1*np.ones((len(order_book['bids']), 1))
		ask_sign = np.ones((len(order_book['asks']), 1))

		# Concatenate the bids and asks with their respective labels.
		bids_with_sign = np.concatenate((bids, bid_sign), axis=1)
		asks_with_sign = np.concatenate((asks, ask_sign), axis=1)

		# Rotate and flip bids and asks so they can be concatenated as one array.
		# This puts the array in ascending order by price.
		bids_with_sign = np.flipud(np.rot90(bids_with_sign, 3))
		asks_with_sign = np.rot90(asks_with_sign, 1)

		# Concatenate the bids and asks.
		observation = np.concatenate((bids_with_sign, asks_with_sign), axis=1)

		# Return the concatenated array of bids and asks.
		# Also return the bid-ask spread.
		return observation, bid, ask, spread

	def _step(self, action):
		"""
		Run one timestep of the environment's dynamics. When end of
		episode is reached, you are responsible for calling `reset()`
		to reset this environment's state.
		Accepts an action and returns a tuple (observation, reward, done, info).
		Args:
			action (object): an action provided by the environment
		Returns:
			observation (object): agent's observation of the current environment
			reward (float) : amount of reward returned after previous action
			done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
			info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
		"""
		self.state, bid, ask, spread = self._observe()

		# The score is the total value of holdings, using BTC as the reference.
		reward = None

		done = None

		info = {}

		return self.state, reward, done, info

	def _reset(self):
		"""
		Resets the state of the environment and returns an initial observation.

		Returns: observation (object): the initial observation of the space. This
		is an array representing the order book.
		"""
		self.state = self._observe()[0]
		return self.state

	def _render(self, mode='human', close=False):
		"""
		Renders the environment.
		The set of supported modes varies per environment. (And some
		environments do not support rendering at all.) By convention,
		if mode is:
		- human: render to the current display or terminal and
		  return nothing. Usually for human consumption.
		- rgb_array: Return an numpy.ndarray with shape (x, y, 3),
		  representing RGB values for an x-by-y pixel image, suitable
		  for turning into a video.
		- ansi: Return a string (str) or StringIO.StringIO containing a
		  terminal-style text representation. The text can include newlines
		  and ANSI escape sequences (e.g. for colors).
		Note:
			Make sure that your class's metadata 'render.modes' key includes
			  the list of supported modes. It's recommended to call super()
			  in implementations to use the functionality of this method.
		Args:
			mode (str): the mode to render with
			close (bool): close all open renderings
		Example:
		class MyEnv(Env):
			metadata = {'render.modes': ['human', 'rgb_array']}
			def render(self, mode='human'):
				if mode == 'rgb_array':
					return np.array(...) # return RGB frame suitable for video
				elif mode is 'human':
					... # pop up a window and render
				else:
					super(MyEnv, self).render(mode=mode) # just raise an exception
		"""
		if mode == 'ansi':
			raise NotImplementedError
		elif mode == 'human':
			raise NotImplementedError
		else:
			super(Market, self).render(mode=mode) # raise an exception

	def _close(self):
		"""
		Override _close in your subclass to perform any necessary cleanup.
		Environments will automatically close() themselves when
		garbage collected or when the program exits.
		"""
		pass

	def _seed(self, seed=None):
		"""
		Sets the seed for this env's random number generator(s).
		Note:
			Some environments use multiple pseudorandom number generators.
			We want to capture all such seeds used in order to ensure that
			there aren't accidental correlations between multiple generators.
		Returns:
			list<bigint>: Returns the list of seeds used in this env's random
			  number generators. The first value in the list should be the
			  "main" seed, or the value which a reproducer should pass to
			  'seed'. Often, the main seed equals the provided 'seed', but
			  this won't be true if seed=None, for example.
		"""
		self.np_random, seed = seeding.np_random(seed)
		return [seed]