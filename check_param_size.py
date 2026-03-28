from dqn_cnn import DQN_CNN

model = DQN_CNN(num_actions=3)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(total_params)