import matplotlib.pyplot as plt

def visualize_q_values(q_values_history, loss_history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(q_values_history, color='b', label='Average Q-Values')
    plt.xlabel('Episode')
    plt.ylabel('Q-Value')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(loss_history, color='r', label='Average Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('train_plot.png')
    plt.close()
    print('q-value 시각화 완료')      