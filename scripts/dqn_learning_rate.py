'''
Description: 
This script defines a Deep Q-Network (DQN) agent for reinforcement learning tasks.
The DQNAgent class encapsulates the functionality for learning from experiences, 
selecting actions, and adjusting the learning rate based on the training process.

Author: LYD
Date: 2024-12-01 17:00:25
'''
import torch
import torch.optim as optim
from collections import deque
import random
import torch
import torch.optim as optim
from collections import deque 
import random

# 定义DQN智能体类
class DQNAgent:
    # 初始化方法，设置智能体的参数
    def __init__(self, input_dim, output_dim, lr=1e-4, device='cpu'):
        self.input_dim = input_dim  # 输入层的维度
        self.output_dim = output_dim  # 输出层的维度
        self.lr = lr  # 学习率
        self.device = device  # 指定设备，如CPU或GPU
        self.memory = deque(maxlen = 5000)  # 经验回放池，最大长度为5000
        self.guided_memory = deque(maxlen = 1000) # 引导数据池
        self.gamma = 0.99  # 奖励折扣因子
        self.epsilon = 0.9  # 用于随机策略的参数
        self.epsilon_min = 0.1  # 随机策略epsilon的最小值
        self.epsilon_decay = 0.9  # 随机策略可能性epsilon的衰减率

        self.state = [] # 状态列表
        self.action = [] # 动作列表

        # 构建模型并将其发送到指定设备
        self.model = self.build_model().to(self.device)
        # 使用Adam优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def build_model(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 128),  # 输入层到第一个隐藏层的线性变换
            torch.nn.ReLU(),  # 第一个隐藏层的激活函数
            torch.nn.Linear(128, 256),  # 第一个隐藏层到第二个隐藏层的线性变换
            torch.nn.ReLU(),  # 第二个隐藏层的激活函数
            torch.nn.Linear(256, 512),  # 第二个隐藏层到第三个隐藏层的线性变换
            torch.nn.ReLU(),  # 第三个隐藏层的激活函数
            torch.nn.Dropout(0.5),  # Dropout层，减少过拟合
            torch.nn.Linear(512, 512),  # 第三个隐藏层到第四个隐藏层的线性变换
            torch.nn.ReLU(),  # 第四个隐藏层的激活函数
            torch.nn.Dropout(0.5),  # 另一个Dropout层
            torch.nn.Linear(512, self.output_dim)  # 第四个隐藏层到输出层的线性变换
    )
        return model

    # 根据当前状态选择动作
    def select_action(self, state):
        if random.random() < self.epsilon:  # 以epsilon的概率随机选择动作
            print("随机策略:")
            return random.randint(0, self.output_dim - 1)
        else:  # 否则选择模型预测的最佳动作
            with torch.no_grad():  # 不计算梯度
                print("Agent选择:")
                return torch.argmax(self.model(torch.tensor(state, dtype=torch.float32).to(self.device))).item()

    # 存储经验到回放池
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if reward > 0.5:  # 将奖励较高的经验存入引导数据池
            self.guided_memory.append((state, action, reward, next_state, done))

    # 从回放池中学习
    def learn(self):
        if len(self.memory) < 64:
            return

        # 随机选择数据
        if random.random() < 0.5:
            if len(self.guided_memory) >= 32:
                # 50%的概率从引导数据池和随机数据池中各取32条数据
                guided_batch = random.sample(self.guided_memory, min(32, len(self.guided_memory)))
                random_batch = random.sample(self.memory, 32)
                batch = guided_batch + random_batch
            else :
                batch = random.sample(self.memory, 64)
        else:
            # 50%的概率完全随机选择64条经验
            batch = random.sample(self.memory, 64)
            
        states, actions, rewards, next_states, dones = zip(*batch)

        # 将states列表转换为张量
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        states = states.view(-1, self.input_dim)  # 重塑张量形状

        # 将actions列表转换为张量
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)

        # 将rewards列表转换为张量
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        # 将next_states列表转换为张量
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        next_states = next_states.view(-1, self.input_dim)  # 重塑张量形状

        # 将dones列表转换为张量
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # 使用Q-learning更新规则
        q_values = self.model(states)
        next_q_values = self.model(next_states)

        target = rewards + self.gamma * torch.max(next_q_values, dim=1)[0] * (1 - dones)
        expected_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = torch.nn.functional.mse_loss(expected_q_values, target)

        self.optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新参数

        # 更新epsilon值
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # 基于训练损失调整学习率
    def adjust_learning_rate(self, reward, next_state, optimizer):
        # 拿出上一步的动作和上一步的state
        if self.state and self.action is not None:
            state = self.state.pop()
            action = self.action.pop()
        else:
            state = 0
            action = 0
        # 存储经验
        self.store_experience(state, action, reward, next_state, done=False)
        # 将next_state压入到state列表里
        self.state.append(next_state)
        # 基于当前state选择动作
        action = self.select_action([next_state])  
        self.action.append(action)
        # 根据动作调整学习率
        if action == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9  # 减小学习率
                print(f"学习率减小，调整为: {param_group['lr']}")
        elif action == 2:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 1.1  # 增大学习率
                print(f"学习率增大，调整为: {param_group['lr']}")
        elif action == 1:
            print("学习率不变。")
        # 学习
        self.learn()