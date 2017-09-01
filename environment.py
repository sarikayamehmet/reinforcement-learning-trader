import ccxt

from gym.core import Env

class Market(Env):
	def __init__(self, exchange = ccxt.bittrex()):
		pass