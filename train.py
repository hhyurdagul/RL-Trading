from data_processor import DataProcessor
from trading_env import TradingEnvironment
from sb3_contrib import RecurrentPPO


def main():
    # Process data
    data_processor = DataProcessor("data/XAUUSD_M15.csv")
    df = data_processor.prepare_data()

    df = df[(df.index >= "2019-01-01") & (df.index < "2024-01-01")]

    env = TradingEnvironment(df, initial_balance=100000)
    # Initialize the agent
    model = RecurrentPPO(
        "MlpLstmPolicy",
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
    model.save("models/model.zip")


if __name__ == "__main__":
    main()