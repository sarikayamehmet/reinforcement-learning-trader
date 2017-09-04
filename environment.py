import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Box

import numpy as np
import ccxt

class OrderSpace(gym.Space):
	"""
	Action space for the Market environment.

	- place order (binary): place order or do nothing
	- type (binary): 'market' or 'limit'
	- side (binary): 'buy' or 'sell'
	- amount (float): how much to trade (could be in base or quote, check API)
	- price (float): percentage of current market price (for limit orders only)

	The amount must be constrained to what I hold in either the base or
	quote currency. This is the max_amount.

	The highest price on the order book will be the max_price. Should this
	update over time? Seems like a bad idea to have a changing control space.

	TODO: Also need to build in certain limits.
	- Cannot place a limit sell order lower than the market price.
	- Cannot place a limit buy order higher than the market price.

	How do I do this without changing the control space? Should I normalize
	the price so that it is a percentage between -100 percent and 100 percent
	of the current price?

	If limit buy, [-1, 0)
	If limit sell, (0, 1]
	If market order, 0

	Or more simply, (0, 1]. Multiply by 1 if sell, by -1 if buy.

		side_sign[side]*(1-prng.np_random.random())

	With this system, no need for max_price.

	Can the amount also be set this way? As a proportion of my holdings in the
	base or quote currency? Intuition says no, since the base amount and quote
	amount can be different so the same percentage can represent different
	amounts when buying or selling.

	TODO: Need to add cancelling an existing order to the control space.
	"""

	side_sign = {'buy': -1, 'sell': 1}

	def __init__(self, max_amount):
		self.max_amount = max_amount

	def sample(self):
		"""
		Uniformly randomly sample a random element of this space
		"""
		self.place_order = np.random.choice([True, False])
		self.type = np.random.choice(['market', 'limit'])
		self.side = np.random.choice(['buy', 'sell'])
		self.amount = np.random.uniform(low=0, high=self.max_amount)
		self.price = self.side_sign[self.side]*(1-np.random.random_sample())

		return [self.place_order, self.type, self.side, self.amount, self.price]

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

class Market(gym.Env):
	"""
	The main OpenAI Gym class. It encapsulates an environment with
	arbitrary behind-the-scenes dynamics. An environment can be
	partially or fully observed.
	The main API methods that users of this class need to know are:
		step
		reset
		render
		close
		seed
	When implementing an environment, override the following methods
	in your subclass:
		_step
		_reset
		_render
		_close
		_seed
	And set the following attributes:
		action_space: The Space object corresponding to valid actions
		observation_space: The Space object corresponding to valid observations
		reward_range: A tuple corresponding to the min and max possible rewards
	Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
	The methods are accessed publicly as "step", "reset", etc.. The
	non-underscored versions are wrapper methods to which we may add
	functionality over time.

	TODO: Look at the mountain_car and continuous_mountain_car envs for reference.
	"""

	metadata = {
		'render.modes': ['human', 'ansi']
	}

	# Set the reward range.
	# TODO: add a multiplier to negative rewards to penalize losses?
	reward_range = (-np.inf, np.inf)

	def __init__(self, exchange, symbol):
		# Load the cryptocurrency exchange.
		self.exchange = exchange
		self.markets = exchange.load_markets()
		self.symbol = symbol

		# Set the max amount (in BTC) per trade.
		# TODO: figure out what to set for max_amount
		self.max_amount = 1.0

		# Action and observation spaces.
		action_space = OrderSpace(max_amount=self.max_amount)
		
		"""
		The observation space is just the order book.
		Bids are an arbitrarily-long list of entries between [0.0, 0.0] and [np.inf, np.inf]
		Asks are an arbitrarily-long list of entries between [0.0, 0.0] and [np.inf, np.inf]

		- order book [[float (0, inf), float (0, inf), int (0, inf)]
		- market price from orderbook? or do I need the VWAP?

		orderbook = exchange.fetch_order_book (exchange.symbols[0])
		bid = orderbook['bids'][0][0] if len (orderbook['bids']) > 0 else None
		ask = orderbook['asks'][0][0] if len (orderbook['asks']) > 0 else None
		spread = (ask - bid) if (bid and ask) else None
		print (exchange.id, 'market price', { 'bid': bid, 'ask': ask, 'spread': spread })
		"""
		observation_space = Box(np.array([0,0.0,0.0]), np.array([np.inf, np.inf, np.inf]))

		# Set the seed for the environment's random number generator.
		self.seed()

		# Reset the environment.
		print self.reset()

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
		pass

	def _reset(self):
		"""
		Resets the state of the environment and returns an initial observation.
		Returns: observation (object): the initial observation of the space.
		
		TODO: return an array representing the order book
		"""
		order_book = self.exchange.fetch_order_book(self.symbol)
		# FIXME: figure out how to set the orderbook as one array
		# - reverse order of bids and concatenate with asks?
		# - how to make sure system understands difference between bids and asks?
		self.state = np.array(order_book['bids'], order_book['asks'])
		return np.array(self.state)

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