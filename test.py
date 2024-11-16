from sb3_contrib import RecurrentPPO
from trading_env import TradingEnvironment
from data_processor import DataProcessor
import pandas as pd
import numpy as np

model = RecurrentPPO.load("models/model.zip")

def get_data():
    data_processor = DataProcessor("data/XAUUSD_M15.csv")
    df = data_processor.prepare_data()
    df = df[df.index >= "2024-01-01"]
    return df


# Seperate data into 2-Week periods
def separate_data(df):
    sep_date = pd.Timedelta(days=14)
    start_date = df.index[0]
    end_date = start_date + sep_date
    periods = []
    while end_date <= df.index[-1]:
        periods.append(df[start_date:end_date].iloc[:-1])
        start_date = end_date
        end_date = start_date + sep_date
    periods.append(df[start_date:])
    return periods


def get_total_profit(df):
    env = TradingEnvironment(df)
    obs = env.reset()[0]
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info, _ = env.step(action)
        if env.total_profit <= -1000 or env.total_profit >= 4000:
            done = True

    return round(env.total_profit)

def get_total_profit_recurrent(df):
    env = TradingEnvironment(df)
    obs = env.reset()[0]
    done = False
    lstm_states = None
    episode_start = np.ones((0,), dtype=bool)
    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
        obs, rewards, done, info, _ = env.step(action)
        episode_start = done
        if env.total_profit <= -1000 or env.total_profit >= 40000:
            done = True

    return round(env.total_profit)

def run_test():
    data = separate_data(get_data())
    total_profits = []
    for df in data:
        total_profits.append(get_total_profit_recurrent(df))
    return total_profits


if __name__ == "__main__":
    result = run_test()
    print(result)
    print(f"Sum: {sum(result)}")

