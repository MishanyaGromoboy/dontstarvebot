from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
import random
from gym_donstarve.donstarve_env import DonStarveEnv
# Инициализируйте игру Don't Starve Together
env = "D:/Steam/steamapps/common/Don't Starve Together"

# Определите размерность состояния игры
state_size = (128, 128, 1)

# Определите количество возможных действий в игре
action_size = 3

# Определите размер пакета и количество эпизодов для обучения
batch_size = 32
n_episodes = 1000

# Определите гиперпараметры для агента
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
memory_size = 1000000
pretrain_length = batch_size
target_update_frequency = 10000
from gym.envs.registration import register

register(
    id='DonStarveEnv-v0',
    entry_point='donstarve_env:DonStarveEnv',
)


# Создайте класс агента DQNAgent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.state_size))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(next_state)[0])
                target[0][action] = reward + self.gamma * Q_future
            self.model.fit(state, target, epochs=1, verbose=0)


import numpy as np
from collections import deque
from dqn_agent import DQNAgent

# Инициализируйте игру Don't Starve Together
env = DonStarveEnv('D:/Steam/steamapps/common/Don\'t Starve Together')

# Определите размерность состояния игры
state_size = env.observation_space.shape

# Определите количество возможных действий в игре
action_size = env.action_space.n

# Инициализируйте агента
agent = DQNAgent(state_size, action_size)

# Определите параметры обучения
batch_size = 32
n_episodes = 1000
max_steps = 1000

# Инициализируйте очередь для сохранения предыдущих состояний игры
prev_states = deque(maxlen=4)

for e in range(n_episodes):
    # Сбросьте состояние игры
    state = env.reset()
    # Обнулите счетчик шагов
    step = 0
    # Обнулите счетчик награды
    total_reward = 0
    # Сбросьте очередь предыдущих состояний игры
    prev_states.clear()
    # Добавьте начальное состояние игры в очередь
    for i in range(4):
        prev_states.append(np.zeros(state_size))

    while True:
        # Получите текущее состояние игры
        current_state = state.reshape(1, *state_size)
        # Добавьте текущее состояние игры в очередь
        prev_states.append(current_state)
        # Получите состояние игры, состоящее из предыдущих четырех кадров
        stacked_states = np.concatenate(prev_states, axis=3)
        # Выберите действие, используя стратегию агента
        action = agent.act(stacked_states)
        # Примените выбранное действие и получите следующее состояние игры и награду
        next_state, reward, done, _ = env.step(action)
        # Добавьте награду к общей награде
        total_reward += reward
        # Получите следующее состояние игры
        next_state = next_state.reshape(1, *state_size)
        # Добавьте следующее состояние игры в очередь
        prev_states.append(next_state)
        # Получите состояние игры, состоящее из предыдущих четырех кадров
        next_stacked_states = np.concatenate(prev_states, axis=3)
        # Добавьте состояние игры, действие, награду и следующее состояние игры в память агента
        agent.remember(stacked_states, action, reward, next_stacked_states, done)
        # Обучите агента на батче из памяти
        agent.learn(batch_size)
        # Обновите текущее состояние игры
        state = next_state



