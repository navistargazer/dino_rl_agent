import matplotlib.pyplot as plt
import os

def draw_plots(history_q_values, history_td_error, history_loss, history_survived, plot_path):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(history_q_values, color='b', label='Fatal_Q-Value')
    plt.xlabel('Episode')
    plt.ylabel('Q-Value')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(history_td_error, color='b', label='AVG TD-Errors')
    plt.xlabel('Episode')
    plt.ylabel('TD-Error')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(history_loss, color='b', label='AVG Losses')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(history_survived, color='r', label='Survived MA10')
    plt.xlabel('Episode')
    plt.ylabel('Time Survived')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()