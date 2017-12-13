import ccxt

from agent import DRQN
from environment import Market
import config

class Trader(object):
	"""
	Class that encapsulates both the environment and the agent.
	"""
	def __init__(self):
		self.market = Market(ccxt.bittrex({
			'apiKey': config.api_key,
			'secret': config.secret,
		}), 'XMR/BTC')
		self.agent = DRQN()
		self.live = False

	def trade(self):
		self.agent.train(n_epochs=2000)
		self.agent.play(n_epochs=100)


def main():
	"""
	Load the market and pass it into the agent.
	"""
	trader = Trader()
	trader.trade()


if __name__ == "__main__":
	main()
