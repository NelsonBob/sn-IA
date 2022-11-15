import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
import matplotlib.pyplot as plt
from IPython import display



MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001



class Agent:
    

    def __init__(self):
        self.n_games = 0
        self.gamma = 0.5 # discount rate
        self.epsilon = 0 # aléatoire
        self.model = Linear_QNet(11, 256, 3)
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger droit
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger à droite
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger à gauche
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # emplacement de l'objectif
            game.food.x > game.head.x,  # objectif à droite
            game.food.x < game.head.x,  # objectif à  gauche
            game.food.y > game.head.y  # objectif en bas
            game.food.y < game.head.y,  # objectif en haut
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # liberer le premier si overflow (MAX_MEMORY)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # liste des tuples
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # déplacement aléatoire
        self.epsilon = 80 - self.n_games
        mouvement_final = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            mouvement = random.randint(0, 2)
            mouvement_final[mouvement] = 1
        else:
            etat0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(etat0)
            mouvement = torch.argmax(prediction).item()
            mouvement_final[mouvement] = 1
        return mouvement_final

plt.ion()

    
def plot( scores, mean_scores):
    display.clear_output(wait=True)
    plt.clf()
    plt.title('Entrainement...')
    plt.xlabel('Nombre de jeux')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # ancien état
        state_old = agent.get_state(game)

        # mouvement
        final_move = agent.get_action(state_old)

        # executer le mouvement et avoir le nouvel état
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # entrainement court
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # entrainement long, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()


            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)



if __name__ == '__main__':
    train()




