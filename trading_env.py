import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TradingEnvironment(gym.Env):
    def __init__(
        self,
        df,
        initial_balance=10000,
        transaction_fee_percent=0.001,
        holding_cost=0.001,
    ):
        super(TradingEnvironment, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.holding_cost = holding_cost  # Cost for holding positions overnight
        self.transaction_fee_cost = 0.2

        # Action space: 0 (Close & Sell), 1 (Hold), 2 (Close & Buy)
        self.action_space = spaces.Discrete(3)

        # Observation space: price data + technical indicators + account info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 1 for long, -1 for short, 0 for no position
        self.position_size = 100  # Amount of contracts
        self.entry_price = 0
        self.current_price = self.df.iloc[self.current_step]["Close"]
        self.trades = []
        self.total_profit = 0
        self.current_value = self.balance
        self.holding_ticks = 0

        return self._get_observation(), {}

    def _get_observation(self):
        features = self.df.iloc[self.current_step]

        # Pre-calculate position value for efficiency
        position_value = self._calculate_position_value() if self.position != 0 else 0

        # Use numpy operations for better performance
        obs = np.concatenate(
            [
                features.to_numpy(),
                np.array(
                    [
                        self.balance,
                        self.position,
                        position_value,
                        self.holding_ticks,
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

        return obs

    def _calculate_position_value(self):
        if self.position != 0:
            return (
                self.position_size
                * (self.current_price - self.entry_price)
                * self.position
            )
        return 0.0

    def step(self, action):
        self.current_price = self.df.iloc[self.current_step]["Close"]

        # Initialize reward
        reward = 0

        # Apply holding cost if position is open
        # If the current day is changed
        if self.position != 0 and self.df.index[self.current_step].day != self.position_time.day:
            holding_penalty = abs(
                self.position_size * self.current_price * self.holding_cost
            )
            reward -= holding_penalty

        # Execute trading action
        if action == 0:  # Close current position (if any) and Sell
            # Close any existing position
            if self.position != 0:
                position_value = self._calculate_position_value()
                self.balance += position_value
                reward += position_value
                self.trades.append(
                    ("Close", self.current_step, self.current_price, position_value)
                )
                self.position = 0

            # Open new short position if no position exists
            elif self.position == 0:
                self.position = -1
                self.entry_price = self.current_price
                self.balance -= self.transaction_fee_cost
                self.holding_ticks = 0
                self.trades.append(("Sell", self.current_step, self.current_price, 0))
                self.position_time = self.df.index[self.current_step]

        elif action == 2:  # Close current position (if any) and Buy
            # Close any existing position
            if self.position != 0:
                position_value = self._calculate_position_value()
                self.balance += position_value
                reward += position_value
                self.trades.append(
                    ("Close", self.current_step, self.current_price, position_value)
                )
                self.position = 0

            # Open new long position if no position exists
            elif self.position == 0:
                self.position = 1
                self.entry_price = self.current_price
                self.balance -= self.transaction_fee_cost
                self.holding_ticks = 0
                self.trades.append(("Buy", self.current_step, self.current_price, 0))
                self.position_time = self.df.index[self.current_step]

        # Move to next step
        self.current_step += 1

        # Calculate current value including open positions
        position_value = self._calculate_position_value()
        self.current_value = self.balance + position_value

        # Update total profit
        self.total_profit = self.current_value - self.initial_balance

        # Check if episode is done
        done = self.current_step >= len(self.df) - 1

        # Add position movement reward/penalty
        if not done and self.position != 0:
            next_price = self.df.iloc[self.current_step]["Close"]
            price_change = (next_price - self.current_price) / self.current_price
            position_reward = (
                price_change * self.position_size * self.current_price * self.position
            )
            reward += position_reward

        # Penalize for holding too long
        if self.holding_ticks > 5:  # Increased penalty for holding more than 5 days
            reward -= abs(
                self.position_size * self.current_price * self.holding_cost * 2
            )

        return (
            self._get_observation(),
            reward,
            done,
            False,
            {
                "current_value": self.current_value,
                "total_profit": self.total_profit,
                "trades": self.trades,
                "position": self.position,
                "holding_days": self.holding_ticks,
            }
        )

    def render(self):
        print(f"Step: {self.current_step}")
        print(f"Balance: ${self.balance:.2f}")
        print(
            f'Position: {"Long" if self.position == 1 else "Short" if self.position == -1 else "No Position"}'
        )
        print(f"Entry Price: ${self.entry_price:.2f}")
        print(f"Current Price: ${self.current_price:.2f}")
        print(f"Current Value: ${self.current_value:.2f}")
        print(f"Total Profit: ${self.total_profit:.2f}")
        print(f"Holding Days: {self.holding_ticks}")
        print(f"Position Time: {self.position_time if self.position != 0 else 'N/A'}")
