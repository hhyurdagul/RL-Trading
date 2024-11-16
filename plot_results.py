import json
import matplotlib.pyplot as plt


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
    with open("history.json", "r") as f:
        history = json.load(f)
    env = history[-1]
    plot_results(env)
