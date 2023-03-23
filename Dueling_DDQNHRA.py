#環境の構築はを参考にする
#あるいはminicondaをインストールしてGym-FightingICE_RL_spring/ftg.ymlから環境を作ってください
#現状GPUを用いるように変数を使用しています

import gym
import sys

from gym_fightingice.envs.Machete import Machete
from py4j.java_gateway import JavaGateway
import py4j
import copy
from collections import deque
import math
import random
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from gym_fightingice.envs.Machete import Machete
from py4j.java_gateway import (CallbackServerParameters, GatewayParameters,
                               JavaGateway, get_field)
from py4j.java_gateway import get_field
import wandb#学習結果がリアルタイムでわかるwebツール weight & bias(wandb)


wandb.init(project="RL-spring-seminar")

#優先度付き経験再生のバッファ
class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity
        self.index = 0
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.priorities[0] = 1.0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.index] = transition

        self.priorities[self.index] = self.priorities.max()
        self.index = (self.index + 1) % self.capacity

    def __len__(self):
        return len(self.memory)

    def sample(self,batch_size):
        alpha=0.6
        beta=0.4
        # 現在経験が入っている部分に対応する優先度を取り出し, サンプルする確率を計算
        priorities = self.priorities[: self.capacity if len(self.memory) == self.capacity else self.index]
        priorities = priorities ** alpha
        prob = priorities / priorities.sum()

        # 上で計算した確率に従ってリプレイバッファ中のインデックスをサンプルする
        indices = np.random.choice(len(self.memory), batch_size, p=prob)
        # 学習の方向性を補正するための重みを計算
        weights = (len(self.memory) * prob[indices]) ** (-beta)
        weights = weights / weights.max()

        # 上でサンプルしたインデックスに基づいて経験をサンプルし, (obs, action, reward, next_obs, done)に分ける
        state, action, next_state,reward, done = zip(*[self.memory[i] for i in indices])

        return state,action,next_state,reward,done,indices,weights
    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities + 1e-4







#ネットワーク下層
class Network(nn.Module):
	def __init__(self, inputSize, hiddenSize, outputSize):
		nn.Module.__init__(self)
		self.l1 = nn.Linear(inputSize, hiddenSize)
		self.l2 = nn.Linear(hiddenSize, hiddenSize)
	def forward(self, x):
		x.cuda()
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		return x

#防御ヘッド(Dueling DQN)
class defenceNetwork(nn.Module):
	def __init__(self, inputSize, hiddenSize, outputSize):
		nn.Module.__init__(self)
		# 状態価値
		self.fc_state = nn.Sequential(
			nn.Linear(hiddenSize,hiddenSize),
			nn.ReLU(),
			nn.Linear(hiddenSize,1)
		)
		# アドバンテージ
		self.fc_advantage = nn.Sequential(
		nn.Linear(hiddenSize,hiddenSize),
		nn.ReLU(),
		nn.Linear(hiddenSize, outputSize)
		)

	def forward(self, x):
		x.cuda()
		state_values = self.fc_state(x)
		advantage = self.fc_advantage(x)
		q_value=state_values + advantage - torch.mean(advantage, dim=1, keepdim=True)
		return q_value


#攻撃ヘッド(Dueling DQN)
class offenceNetwork(nn.Module):
	def __init__(self, inputSize, hiddenSize, outputSize):
		nn.Module.__init__(self)
		# 状態価値
		self.fc_state = nn.Sequential(
			nn.Linear(hiddenSize,hiddenSize),
			nn.ReLU(),
			nn.Linear(hiddenSize, 1)
		)
		# アドバンテージ
		self.fc_advantage = nn.Sequential(
		nn.Linear(hiddenSize,hiddenSize),
		nn.ReLU(),
		nn.Linear(hiddenSize, outputSize)
		)

	def forward(self, x):
		x.cuda()
		state_values = self.fc_state(x)
		advantage = self.fc_advantage(x)
		q_value=state_values + advantage - torch.mean(advantage, dim=1, keepdim=True)
		#x = self.l1(x)
		return q_value





class HRADuelDQNNAgent(object):
	#ハイパーパラメータ
	EPS_START = 1.0 
	EPS_END = 0.1 
	EPS_DECAY = 13000  
	GAMMA = 0.9 
	LR = 0.001  
	INPUT_SIZE = 141 
	HIDDEN_SIZE= 80  
	OUTPUT_SIZE = 40  
	BATCH_SIZE = 32  
	Memory_Capacity = 50000  
	UPDATE_TARGET_Q_FREQ = 300
	steps_done=0
	steps_learn_in_a_round=0
	model_count_in_a_round =0
	use_cuda = True
	# if gpu is to be used
	#use_cuda = torch.cuda.is_available()
	FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
	LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
	ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
	Tensor = FloatTensor
	energy=torch.tensor(np.array([0., 0.03333333, 0.16666667, 0., 0.01666667,0.06666667, 0., 0.03333333, 0.33333333, 0.13333333,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,
	0., 0., 0., 0., 0.01666667,0.01666667, 0.06666667, 0.5, 0., 0.18333333,0., 0., 0., 0.01666667, 0.03333333]))#エネルギーが足りない技を使用させないようにするための値

	defence_memory =PrioritizedReplayBuffer(Memory_Capacity)
	offence_memory =PrioritizedReplayBuffer(Memory_Capacity)
	# body
	model = Network(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).cuda()
	target_model = copy.deepcopy(model).cuda()
	defencehead = defenceNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).cuda()
	target_defencehead = copy.deepcopy(defencehead).cuda()
	offencehead = offenceNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).cuda()
	target_offencehead = copy.deepcopy(offencehead).cuda()

	# optimizer
	optimizer = optim.Adam(model.parameters(), LR)
	defenceoptimizer = optim.Adam(defencehead.parameters(), LR)
	offenceoptimizer = optim.Adam(offencehead.parameters(), LR)
	
	#loss
	loss_func = nn.SmoothL1Loss(reduction='none')



	#行動決定
	def select_action(self, state):
		sample = random.random()
		decay =  (self.EPS_START - self.EPS_END) / self.EPS_DECAY
		eps_threshold =self.EPS_START - (decay * self.steps_done)
		if eps_threshold<0.1:#ε-greedy
			eps_threshold=0.1
		self.steps_done+=1
		if sample > eps_threshold:
			self.model_count_in_a_round += 1
			defenceQvalues = self.defencehead(self.model(Variable(state, volatile=True).type(self.FloatTensor))).data
			offenceQvalues = self.offencehead(self.model(Variable(state, volatile=True).type(self.FloatTensor))).data
			Qvalues=defenceQvalues+offenceQvalues
			for i in range(40):#エネルギーが足りないときは実行しないようQ値を極めて小さくする
				if state[0][1]<self.energy[i]:	
					Qvalues[0][i]=-float('inf')
			max_totalQ_index = (Qvalues).max(1)[1].view(1,1)

			return max_totalQ_index
		else:
			x=np.array([1 for i in range(40)])
			y=np.array([])
			for i in range(40):#エネルギーが足りないときは他の行動からランダムに選択
				if state[0][1]>=self.energy[i]:
					y=np.append(y,i)
			action=np.random.choice(y,1)
			return self.LongTensor([[action[0]]])



	def pushdefenceData(self,state,action, next_state, defenceReward,done):
		self.defence_memory.push((self.FloatTensor([state]),action, self.FloatTensor([next_state]), self.FloatTensor([defenceReward]),done))
	def pushoffenceData(self,state,action, next_state,offenceReward,done):
		self.offence_memory.push((self.FloatTensor([state]),action, self.FloatTensor([next_state]),self.FloatTensor([offenceReward]),done))

	def learn(self,episode):
		if len(self.defence_memory) < self.BATCH_SIZE:#バッチサイズ未満のときは何もしない
			return 0.0,0.0,0.0
		self.steps_learn_in_a_round += 1

		
		#defence
		batch_defence_state, batch_defence_action, batch_defence_next_state, batch_defence_reward,defence_done,defence_indices,defence_weights  = self.defence_memory.sample(self.BATCH_SIZE)

		batch_defence_state = Variable(torch.cat(batch_defence_state))
		batch_defence_action = Variable(torch.cat(batch_defence_action))
		batch_defence_reward = Variable(torch.cat(batch_defence_reward))
		batch_defence_next_state = Variable(torch.cat(batch_defence_next_state))

		# current Q values are estimated by NN for all actions of defence
		current_q_values = self.model(batch_defence_state)

		
		current_defenceq_values=self.defencehead(current_q_values).gather(1, batch_defence_action)

		max_next_defence_current_q_values_action = torch.argmax(self.defencehead(self.model(batch_defence_next_state)).detach(),dim=1)#Double DQN用の行動を取る
		next_defenceq_values = (self.target_defencehead(self.target_model(batch_defence_next_state)).detach()).gather(1, max_next_defence_current_q_values_action.unsqueeze(1)).squeeze(1)
		expected_defenceq_values = batch_defence_reward + (1-torch.tensor(np.array(defence_done).astype(np.int),device="cuda:0"))*(self.GAMMA * next_defenceq_values)
		# loss is measured from error between current and newly expected Q values
		defence_weights=torch.tensor(defence_weights,device="cuda:0")
		defenceLoss = (defence_weights *self.loss_func(current_defenceq_values.squeeze(), expected_defenceq_values)).mean()
		self.defence_memory.update_priorities(defence_indices, (expected_defenceq_values - current_defenceq_values.squeeze(1)).abs().detach().cpu().numpy())

		#attack
		batch_offence_state, batch_offence_action, batch_offence_next_state, batch_offence_reward,offence_done,offence_indices,offence_weights  = self.offence_memory.sample(self.BATCH_SIZE)

		batch_offence_state = Variable(torch.cat(batch_offence_state))
		batch_offence_action = Variable(torch.cat(batch_offence_action))
		batch_offence_reward = Variable(torch.cat(batch_offence_reward))
		batch_offence_next_state = Variable(torch.cat(batch_offence_next_state))

		current_offenceq_values=self.offencehead(current_q_values).gather(1, batch_offence_action)

		max_next_offence_current_q_values_action = torch.argmax(self.offencehead(self.model(batch_offence_next_state)).detach(),dim=1)#Double DQN用の行動を取る
		next_offenceq_values =(self.target_offencehead(self.target_model(batch_offence_next_state)).detach()).gather(1, max_next_offence_current_q_values_action.unsqueeze(1)).squeeze(1)
		expected_offenceq_values = batch_offence_reward + (1-torch.tensor(np.array(offence_done).astype(np.int),device="cuda:0"))*(self.GAMMA * next_offenceq_values)
		# loss is measured from error between current and newly expected Q values
		offence_weights=torch.tensor(offence_weights,device="cuda:0")
		offenceLoss = (offence_weights *self.loss_func(current_offenceq_values.squeeze(), expected_offenceq_values)).mean()
		self.offence_memory.update_priorities(offence_indices, (expected_offenceq_values - current_offenceq_values.squeeze(1)).abs().detach().cpu().numpy())

		Loss=defenceLoss+offenceLoss




		self.optimizer.zero_grad()
		self.defenceoptimizer.zero_grad()
		self.offenceoptimizer.zero_grad()
		Loss.backward()
		self.optimizer.step()
		self.defenceoptimizer.step()
		self.offenceoptimizer.step()


		torch.save(self.target_model.state_dict(),'/home/t-yamamoto/Desktop/DRL_SPS_Fighting/Gym-FightingICE_RL_spring/model1/network'+str(episode)+'.pth')#保存場所はお使いの端末に合わせてください
		torch.save(self.target_defencehead.state_dict(),'/home/t-yamamoto/Desktop/DRL_SPS_Fighting/Gym-FightingICE_RL_spring/model2/defence'+str(episode)+'.pth')
		torch.save(self.target_offencehead.state_dict(),'/home/t-yamamoto/Desktop/DRL_SPS_Fighting/Gym-FightingICE_RL_spring/model3/offence'+str(episode)+'.pth')

		# ターゲットネットワークの更新
		if self.steps_learn_in_a_round % self.UPDATE_TARGET_Q_FREQ == 0:
			self.target_model = copy.deepcopy(self.model)
			self.target_defencehead = copy.deepcopy(self.defencehead)
			self.target_offencehead= copy.deepcopy(self.offencehead)

		return Loss,defenceLoss,offenceLoss



def main():
	#学習開始時の値の初期化
	sys.path.append('gym-fightingice')
	episodes=1500
	agent=HRADuelDQNNAgent()
	total_win=0
	timestep=0
	HP=10000.0

	for episode in range(episodes):
		#episode開始時の値初期化
		if episode==0:
			env = gym.make("FightingiceDataFrameskip-v0", java_env_path="", port=5000)
		nostate=env.reset(p2=Machete)
		state=nostate[0][0]		
		done=False
		epireward=0.0
		ableaction=True
		before_energy=0

		while not done:
			action = agent.select_action(agent.FloatTensor([state]))
			wandb.log({'timestep':timestep, 'action':action})#wandbに行動を記録
			next_state,dummy_reward,done,info=env.step(int(action))
			decay =  (agent.EPS_START - agent.EPS_END) / agent.EPS_DECAY
			eps_threshold = agent.EPS_START - (decay * agent.steps_done)
			if eps_threshold<0.1:
				eps_threshold =0.1
			wandb.log({'timestep':timestep, 'epsilon':eps_threshold})#wandbにεを記録
			timestep+=1
			innext_state=next_state[0]
			defencereward=HP*(np.array(innext_state)[0]-np.array(state)[0])#報酬計算(自分のダメージ)
			offencereward=HP*(np.array(state)[64]-np.array(innext_state)[64])#報酬計算(相手のダメージ)
			if done==True:
				tomydamage=1.0-np.array(innext_state)[0]
				tooppdamage=1.0-np.array(innext_state)[64]
				score=1000*tooppdamage/(tomydamage+tooppdamage)#スコアの計算
				defencereward+=score-500
				offencereward+=score-500

			epireward+=defencereward+offencereward

			agent.pushdefenceData(state,action,innext_state, defencereward,done)
			agent.pushoffenceData(state,action,innext_state,offencereward,done)

			state =innext_state
			all_loss,defenceLoss,offenceLoss=agent.learn(episode)
			wandb.log({'timestep':timestep, 'all_loss':float(all_loss)})
			state =innext_state
		if np.array(innext_state)[0]>np.array(innext_state)[64]:#体力が上回っていれば勝利数+1
			total_win+=1
		tomydamage=1.0-np.array(innext_state)[0]
		tooppdamage=1.0-np.array(innext_state)[64]
		score=1000*tooppdamage/(tomydamage+tooppdamage)#スコアの計算
		wandb.log({'episode': episode, 'score': score})#episodeのスコアをwandbに通知
		wandb.log({'episode': episode, 'total_win': total_win})#episode時点の勝利数をwandbに通知
		wandb.log({'episode': episode, 'epireward': epireward})#episodeの合計報酬をwandbに通知
		wandb.log({'episode': episode, 'my_HP':HP*innext_state[0]})#episodeの自体力をwandbに通知
		wandb.log({'episode': episode, 'opp_HP':HP*innext_state[64]})#episodeの相手体力をwandbに通知





if __name__ == "__main__":
	main()

