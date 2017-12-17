import gym
from gym.utils import seeding

import numpy as np


class OrderSpace(gym.Space):
    """
    Action space for the Market environment.

    - place order (binary): place order or do nothing
    - order type (binary): 'market' or 'limit'
    - side (binary): 'buy' or 'sell'
    - amount_proportion (float): proportion of holdings to trade (could be in base or quote, check the API)
    - price_percentage (float): percentage of current market price (for limit orders only)
    """

    side_sign = {'buy': -1, 'sell': 1}

    def __init__(self):
        pass

    def sample(self):
        """
        Uniformly randomly sample a random element of this space
        """
        self.place_order = np.random.choice([True, False])
        self.order_type = np.random.choice(['market', 'limit'])
        self.side = np.random.choice(['buy', 'sell'])
        self.amount_proportion = np.random.uniform(low=0.0, high=1.0)
        self.price_percentage = self.side_sign[self.side]*(1-np.random.random_sample())

        return [self.place_order, self.order_type, self.side, self.amount_proportion, self.price_percentage]

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


class MarketDataSpace(gym.Space):
    """
    Observation space for the Market environment. (The order book.)

    An arbitrarily long number of columns, where each column has:
    - A discrete variable {-1, 1} indicating a bid or an ask.
    - A continuous variable [0, inf) for the price.
    - A continuous variable [0, inf) for the quantity.
    """

    def __init__(self):
        # self.sample()
        pass

    def sample(self):
        """
        Uniformly randomly sample a random element of this space
        """
        # self.side_sign = np.random.choice([-1, 1])
        # self.price = np.random.uniform(low=0, high=np.inf)
        # self.quantity = np.random.uniform(low=0, high=np.inf)
        raise NotImplementedError

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
        if self.place_order:
            # If it's a market order, create an order without specifying a price.
            if self.order_type == 'market':
                if self.side == 'buy':
                    order_info = self.exchange.create_market_buy_order(self.symbol, self.amount)
                elif self.side == 'sell':
                    order_info = self.exchange.create_market_sell_order(self.symbol, self.amount)
            # Otherwise, include the price in the order.
            elif self.order_type == 'limit':
                if self.side == 'buy':
                    order_info = self.exchange.create_limit_buy_order(self.symbol, self.amount, self.price)
                elif self.side == 'sell':
                    order_info = self.exchange.create_limit_sell_order(self.symbol, self.amount, self.price)
            # Save the order ID returned from placing the order.
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
        self.previous_balance = exchange.fetch_balance()
        self.symbol = symbol

        # Save the starting BTC balance.
        self.starting_BTC = self.previous_balance['BTC']['total']
        # If there's no balance, replace None with 0.
        if self.starting_BTC is None:
            self.starting_BTC = 0

        # Set the goals.
        ## What multiplier should be considered 'success'?
        self.success_metric = 1.01*self.starting_BTC
        ## What multiplier should be considered 'failure'?
        self.failure_metric = 0.8*self.starting_BTC

        # Set the action space. This is defined by the OrderSpace object.
        self.action_space = OrderSpace()

        # Set the observation space. This is defined by the MarketDataSpace object.
        self.observation_space = MarketDataSpace()

        # Set the seed for the environment's random number generator.
        self.seed()

        # Reset the environment.
        self.reset()

    def _observe(self):
        # Fetch the order book (dictionary) for our symbol.
        order_book = self.exchange.fetch_order_book(self.symbol)

        # Calculate the market price.
        bid = order_book['bids'][0][0] if len(order_book['bids']) > 0 else None
        ask = order_book['asks'][0][0] if len(order_book['asks']) > 0 else None
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

        # Return the concatenated array of bids and asks (observation).
        # Also return the bid-ask spread.
        return observation, bid, ask, spread

    def _step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        # Process the action.
        ## Ensure it's a valid action.
        #assert self.action_space.contains(action), "%r (%s) invalid " % (action,type(action))

        ## It should come in the format [place_order, order_type, side, amount, price].
        place_order, order_type, side, amount_proportion, price_percentage = action

        ## Determine the price for the order using the bid-ask spread and the price percentage.
        if side == 'buy':
            price = ask*(1 + price_percentage)
        elif side == 'sell':
            price = bid*(1 + price_percentage)

        ## Determine the amount for the order using the balance and the proportion.
        amount = amount_proportion*(self.previous_balance['BTC']['total']/price)

        ## Create an Order object from the action.
        order = Order(self.exchange, self.symbol, place_order, order_type, side, amount, price)

        ## Place the order.
        order_id = order.place()

        # Observe the state of the environment.
        self.state, bid, ask, spread = self._observe()

        # Calculate the reward.
        ## Fetch the current balance of BTC.
        current_balance = self.exchange.fetch_balance()
        current_BTC = current_balance['BTC']['total']
        ### If there's no balance, replace None with 0.
        if current_BTC is None:
            current_BTC = 0

        ## Get the balance of BTC before this timestep.
        previous_BTC = self.previous_balance['BTC']['total']
        ### If there's no balance, replace None with 0.
        if previous_BTC is None:
            previous_BTC = 0

        ## If the previous BTC balance was 0, the reward is the current BTC balance.
        if previous_BTC == 0:
            reward = current_BTC
        ## Else, calculate the reward by finding the percent change in BTC balance during this timestep.
        else:
            reward = (current_BTC - previous_BTC)/previous_BTC

        # Determine when the episode ends.
        done = False
        ## If the BTC balance drops to the failure metric, end the episode and apply a penalty.
        if current_BTC <= self.failure_metric:
            done = True
            reward -= 1
        ## If the BTC balance rises to the success metric, end the episode and apply a bonus.
        if current_BTC >= self.success_metric:
            done = True
            reward += 1

        # Save diagnostic information for debugging.
        info = {}
        info['order_id'] = order_id
        info['previous_BTC'] = previous_BTC
        info['current_BTC'] = current_BTC

        # Save the current balance for the next step.
        self.previous_balance = current_balance

        # Return the results of the agents action during the timestep.
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
            super(Market, self).render(mode=mode)  # raise an exception

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
