import matplotlib.pyplot as plt

def visualize_q_values(history_reward, history_q_values):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_q_values, color='b', label='Average Q-Values')
    plt.xlabel('Episode')
    plt.ylabel('Q-Value')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history_reward, color='r', label='Total Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('train_plot.png')
    plt.close()