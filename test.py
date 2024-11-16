from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import TradingEnvironment
from data_processor import DataProcessor
import matplotlib.pyplot as plt
import json


def run_test():
    data_processor = DataProcessor("XAUUSD_M15.csv")
    df = data_processor.prepare_data()

    df = df[df.index >= "2024-06-01"].iloc[:20]

    model = PPO.load("xauusd_trading_model")

    env = DummyVecEnv([lambda: TradingEnvironment(df)])
    obs = env.reset()
    done = False
    count = 0
    history = []
    while not done:
        if count % 100 == 0:
            print(f"Step: {count}")
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        count += 1

    history.append(
        {
            "current_value": info[0]["current_value"],
            "total_profit": info[0]["total_profit"],
            "trades": info[0]["trades"],
            "position": info[0]["position"],
            "holding_days": info[0]["holding_days"],
        }
    )
    json.dump(history, open("history.json", "w"))


if __name__ == "__main__":
    run_test()
