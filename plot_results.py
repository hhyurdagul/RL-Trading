import json
import matplotlib.pyplot as plt
import pandas as pd



def func(data):
    # Convert the data into a DataFrame
    df = pd.DataFrame(data["trades"], columns=["Action", "Trade_ID", "Price", "Profit_Loss"])

    # Plot 1: Trade Actions over Trade_ID
    plt.figure(figsize=(12, 6))
    plt.plot(df['Trade_ID'], df['Price'], marker='o', linestyle='-', label='Price')
    plt.scatter(df[df['Action'] == 'Buy']['Trade_ID'], df[df['Action'] == 'Buy']['Price'], color='green', label='Buy', marker='^')
    plt.scatter(df[df['Action'] == 'Sell']['Trade_ID'], df[df['Action'] == 'Sell']['Price'], color='red', label='Sell', marker='v')
    plt.scatter(df[df['Action'] == 'Close']['Trade_ID'], df[df['Action'] == 'Close']['Price'], color='blue', label='Close', marker='x')
    plt.xlabel('Trade ID')
    plt.ylabel('Price')
    plt.title('Trading Actions and Prices')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 2: Profit/Loss per Trade
    plt.figure(figsize=(12, 6))
    plt.bar(df[df['Action'] == 'Close']['Trade_ID'], df[df['Action'] == 'Close']['Profit_Loss'], color='purple')
    plt.xlabel('Trade ID')
    plt.ylabel('Profit/Loss')
    plt.title('Profit/Loss for Each Close Trade')
    plt.grid(True)
    plt.show()


def plot_results(hist):



    """Plot the trading results"""
    plt.figure(figsize=(12, 6))
    plt.plot([t[2] for t in hist["trades"]], label="Trade Prices")
    plt.scatter(
        [t[1] for t in hist["trades"] if t[0] == "Buy"],
        [t[2] for t in hist["trades"] if t[0] == "Buy"],
        marker="^",
        color="g",
        label="Buy",
    )
    plt.scatter(
        [t[1] for t in hist["trades"] if t[0] == "Sell"],
        [t[2] for t in hist["trades"] if t[0] == "Sell"],
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


if __name__ == "__main__":
    with open("logs/history.json", "r") as f:
        history = json.load(f)
    env = history[-1]
    plot_results(env)

    func(env)
