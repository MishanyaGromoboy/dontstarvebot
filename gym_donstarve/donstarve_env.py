import gym
from gym import spaces
from PIL import Image


class DonStarveEnv:
    def __init__(self, game_path):
        self.game_path = game_path
        self.env = gym.make('DonStarveEnv-v0')
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)

    def start_game(self):
        self.env.start()

    def stop_game(self):
        self.env.stop()

    def reset(self):
        obs = self.env.reset()
        obs = self._preprocess_obs(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._preprocess_obs(obs)
        return obs, reward, done, info

    def _preprocess_obs(self, obs):
        obs = Image.fromarray(obs)
        obs = obs.convert('L')  # переводим изображение в черно-белый формат
        obs = obs.resize((84, 84), Image.BICUBIC)  # изменяем размер до 84x84 пикселя
        obs = np.array(obs).reshape(84, 84, 1)
        return obs
env = DonStarveEnv('D:/Steam/steamapps/common/Don\'t Starve Together')
