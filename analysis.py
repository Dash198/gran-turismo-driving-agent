"""
GT Agent Training Analysis
─────────────────────────
Generates diagnostic plots from monitor.csv to detect:
  • Reward trends & learning progress
  • Reward farming (high reward + low steps = exploit)
  • Episode length distribution & termination patterns
  • Phase detection (exploration → exploitation transition)

Usage:
    uv run analysis.py                          # default monitor.csv
    uv run analysis.py --file logs/SAC/monitor.csv
    uv run analysis.py --last 500               # only analyze last N episodes
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(file_path, last_n=None):
    """Load SB3 monitor CSV with metadata header."""
    df = pd.read_csv(file_path, skiprows=1)
    df.columns = [c.strip() for c in df.columns]
    if last_n and last_n < len(df):
        df = df.tail(last_n).reset_index(drop=True)
    df["episode"] = range(len(df))
    df["reward_per_step"] = df["r"] / df["l"].clip(lower=1)
    df["wall_minutes"] = (df["t"] - df["t"].iloc[0]) / 60.0
    return df


def print_stats(df):
    """Print summary statistics to terminal."""
    n = len(df)
    print("\n" + "=" * 55)
    print("  GT AGENT — TRAINING ANALYSIS")
    print("=" * 55)
    print(f"  Episodes:        {n}")
    print(f"  Total Time:      {df['wall_minutes'].iloc[-1]:.0f} min")
    print(f"  Total Steps:     {df['l'].sum():,.0f}")
    print()

    # Reward stats
    print("  ── REWARD ──")
    print(f"  Mean:            {df['r'].mean():+.1f}")
    print(f"  Median:          {df['r'].median():+.1f}")
    print(f"  Best:            {df['r'].max():+.1f}  (ep {df['r'].idxmax()})")
    print(f"  Worst:           {df['r'].min():+.1f}  (ep {df['r'].idxmin()})")
    print(f"  Std:             {df['r'].std():.1f}")
    print()

    # Length stats
    print("  ── EPISODE LENGTH ──")
    print(f"  Mean:            {df['l'].mean():.0f} steps")
    print(f"  Median:          {df['l'].median():.0f} steps")
    print(f"  Max:             {df['l'].max():.0f} steps  (ep {df['l'].idxmax()})")
    print(f"  Min:             {df['l'].min():.0f} steps")
    print()

    # Efficiency
    print("  ── EFFICIENCY ──")
    print(f"  Reward/Step:     {df['reward_per_step'].mean():+.2f}")
    print(f"  Corr (r vs l):   {df['r'].corr(df['l']):.3f}")
    print()

    # Trend (compare first vs last quarter)
    q = max(1, n // 4)
    early = df.head(q)["r"].mean()
    late = df.tail(q)["r"].mean()
    delta = late - early
    arrow = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"
    print(f"  Trend:           {arrow} {delta:+.1f}  (first {q} vs last {q} eps)")

    # Farming detection
    # High reward + short episodes = potential exploit
    short_eps = df[df["l"] < df["l"].quantile(0.25)]
    if len(short_eps) > 10:
        short_avg = short_eps["reward_per_step"].mean()
        long_eps = df[df["l"] > df["l"].quantile(0.75)]
        long_avg = long_eps["reward_per_step"].mean()
        if short_avg > long_avg * 1.5:
            print(f"  ⚠️  FARMING DETECTED: short eps earn {short_avg:.2f}/step vs {long_avg:.2f}/step")
        else:
            print(f"  ✅  No farming (short: {short_avg:.2f}/step, long: {long_avg:.2f}/step)")
    print("=" * 55 + "\n")


def plot_dashboard(df, save_path="training_analysis.png"):
    """Generate a 2x3 diagnostic dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("GT Agent — Training Dashboard", fontsize=16, fontweight="bold")
    fig.patch.set_facecolor("#1a1a1a")

    for ax in axes.flat:
        ax.set_facecolor("#2a2a2a")
        ax.tick_params(colors="#cccccc", labelsize=8)
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")
        ax.title.set_color("#eeeeee")
        for spine in ax.spines.values():
            spine.set_color("#444444")

    # ── 1. REWARD CURVE ──
    ax = axes[0, 0]
    ax.plot(df["episode"], df["r"], alpha=0.15, color="#4488ff", linewidth=0.5)
    window = min(50, max(5, len(df) // 20))
    rolling = df["r"].rolling(window=window, min_periods=1).mean()
    ax.plot(df["episode"], rolling, color="#ff4444", linewidth=2, label=f"MA({window})")
    ax.axhline(0, color="#666666", linewidth=0.5, linestyle="--")
    ax.set_title("Episode Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend(fontsize=8, facecolor="#333333", labelcolor="#cccccc")

    # ── 2. EPISODE LENGTH ──
    ax = axes[0, 1]
    ax.plot(df["episode"], df["l"], alpha=0.15, color="#44cc44", linewidth=0.5)
    rolling_l = df["l"].rolling(window=window, min_periods=1).mean()
    ax.plot(df["episode"], rolling_l, color="#ffaa00", linewidth=2, label=f"MA({window})")
    ax.set_title("Episode Length")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.legend(fontsize=8, facecolor="#333333", labelcolor="#cccccc")

    # ── 3. REWARD vs LENGTH (scatter — farming check) ──
    ax = axes[0, 2]
    scatter = ax.scatter(
        df["l"], df["r"], alpha=0.4, c=df["episode"], cmap="plasma", s=8, edgecolors="none"
    )
    fig.colorbar(scatter, ax=ax, label="Episode #", fraction=0.04)
    ax.set_title("Reward vs Length (Farming Check)")
    ax.set_xlabel("Episode Length (Steps)")
    ax.set_ylabel("Total Reward")
    # Trend line
    if len(df) > 10:
        z = np.polyfit(df["l"], df["r"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df["l"].min(), df["l"].max(), 100)
        ax.plot(x_line, p(x_line), "--", color="#ff6666", alpha=0.7, linewidth=1)

    # ── 4. REWARD DENSITY (reward per step) ──
    ax = axes[1, 0]
    density_window = min(100, max(10, len(df) // 10))
    density_rolling = df["reward_per_step"].rolling(window=density_window, min_periods=1).mean()
    ax.fill_between(df["episode"], density_rolling, alpha=0.3, color="#44ff44")
    ax.plot(df["episode"], density_rolling, color="#44ff44", linewidth=1.5)
    ax.axhline(0, color="#666666", linewidth=0.5, linestyle="--")
    ax.set_title(f"Reward Density (MA({density_window}))")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward / Step")

    # ── 5. LENGTH HISTOGRAM ──
    ax = axes[1, 1]
    ax.hist(df["l"], bins=30, color="#8844ff", alpha=0.7, edgecolor="#aa66ff")
    ax.axvline(df["l"].mean(), color="#ff4444", linestyle="--", linewidth=1.5, label=f"Mean: {df['l'].mean():.0f}")
    ax.axvline(df["l"].median(), color="#ffaa00", linestyle="--", linewidth=1.5, label=f"Median: {df['l'].median():.0f}")
    ax.set_title("Episode Length Distribution")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Count")
    ax.legend(fontsize=7, facecolor="#333333", labelcolor="#cccccc")

    # ── 6. STEPS/SECOND (throughput) ──
    ax = axes[1, 2]
    if len(df) > 1:
        dt = df["t"].diff().clip(lower=0.1)
        throughput = df["l"] / dt
        throughput_roll = throughput.rolling(window=window, min_periods=1).mean()
        ax.plot(df["episode"], throughput_roll, color="#ff88ff", linewidth=1.5)
        ax.axhline(throughput_roll.median(), color="#666666", linestyle="--", linewidth=0.5)
        ax.set_title(f"Throughput — Median: {throughput_roll.median():.1f} steps/s")
    else:
        ax.set_title("Throughput (not enough data)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps / Second")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"📊 Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="GT Agent Training Analysis")
    parser.add_argument("--file", default="logs/SAC/monitor.csv", help="Path to monitor.csv")
    parser.add_argument("--last", type=int, default=None, help="Only analyze last N episodes")
    parser.add_argument("--out", default="training_analysis.png", help="Output plot filename")
    args = parser.parse_args()

    try:
        df = load_data(args.file, args.last)
    except Exception as e:
        print(f"❌ Error loading {args.file}: {e}")
        return

    if len(df) < 5:
        print(f"⚠️  Only {len(df)} episodes — need at least 5 for meaningful analysis.")
        return

    print_stats(df)
    plot_dashboard(df, save_path=args.out)


if __name__ == "__main__":
    main()
