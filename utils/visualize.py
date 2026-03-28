import matplotlib.pyplot as plt
import os

def draw_plots(history_survived, history_q_values):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_q_values, color='b', label='Max Q-Values')
    plt.xlabel('Episode')
    plt.ylabel('Q-Value')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history_survived, color='r', label='Time Survived')
    plt.xlabel('Episode')
    plt.ylabel('Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/train_plot.png')
    plt.close()