from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from data_processor import DataProcessor
from trading_env import TradingEnvironment
import matplotlib.pyplot as plt

def plot_results(env):
    """Plot the trading results"""
    plt.figure(figsize=(12, 6))
    plt.plot([t[2] for t in env.trades], label="Trade Prices")
    plt.scatter(
        [t[1] for t in env.trades if t[0] == "Buy"],
        [t[2] for t in env.trades if t[0] == "Buy"],
        marker="^",
        color="g",
        label="Buy",
    )
    plt.scatter(
        [t[1] for t in env.trades if t[0] == "Sell"],
        [t[2] for t in env.trades if t[0] == "Sell"],
        marker="v",
        color="r",
        label="Sell",
    )
    plt.title("Trading Actions and Price Movement")
    plt.xlabel("Step")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig("trading_results.png")
    plt.close()


def main():
    # Process data
    data_processor = DataProcessor("XAUUSD_M15.csv")
    df = data_processor.prepare_data()

    df = df[(df.index >= "2024-01-01") & (df.index < "2024-06-01")]

    # Create and wrap the environment
    env = DummyVecEnv([lambda: TradingEnvironment(df)])

    # Initialize the agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0001,  # Reduced learning rate for stability
        n_steps=1024,          # Reduced steps for faster updates
        batch_size=128,        # Increased batch size for better parallelization
        n_epochs=5,            # Reduced epochs for faster training
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,        # Reduced entropy coefficient
        policy_kwargs=dict(
            net_arch=dict(
                pi=[64, 64],   # Smaller network architecture
                vf=[64, 64]
            )
        ),
        device="auto"          # Will use GPU if available
    )

    # Train the agent
    print("Training the agent...")
    model.learn(total_timesteps=10000)
    print("Training completed.")

    # Save the trained model
    model.save("xauusd_trading_model")

    # Evaluate the model
    obs = env.reset()
    done = False
    count = 0
    while not done:
        if count % 100 == 0:
            print(f"Step: {count}")
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        count += 1

    # Plot results
    plot_results(env.envs[0])

    print(f"Final portfolio value: ${env.envs[0].current_value:.2f}")
    print(f"Total profit: ${env.envs[0].total_profit:.2f}")
    print(f"Number of trades: {len(env.envs[0].trades)}")





if __name__ == "__main__":
    main()