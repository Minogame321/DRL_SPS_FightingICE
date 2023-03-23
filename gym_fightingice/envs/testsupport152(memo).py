import gym
import sys


from gym_fightingice.envs.Machete import Machete
from py4j.java_gateway import JavaGateway
import py4j
import time

import copy
import gym
from collections import deque
import math
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym_fightingice.envs.Machete import Machete
from gym_fightingice.envs.Forward import Forward
from gym_fightingice.envs.Machete_4 import Machete_4
from gym_fightingice.envs.Machete_3 import Machete_3
from gym_fightingice.envs.Machete_2 import Machete_2
from gym_fightingice.envs.Machete_1 import Machete_1
from gym_fightingice.envs.KickAI import KickAI
from gym_fightingice.envs.NoAI import NoAI
from gym_fightingice.envs.randomAI import randomAI
from py4j.java_gateway import (CallbackServerParameters, GatewayParameters,
                               JavaGateway, get_field)
from py4j.java_gateway import get_field
import wandb

from torch.autograd import Variable



wandb.init(project="research-result-demo")
class ReplayMemory:
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []

	def push(self, transition):
		self.memory.append(transition)
		if len(self.memory) > self.capacity:
			del self.memory[0]
			
	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)
		

class Defence_Network(nn.Module):
	def __init__(self, inputSize, hiddenSize, outputSize):
		nn.Module.__init__(self)
		self.l1 = nn.Linear(inputSize, 128)
		self.l2 = nn.Linear(128, hiddenSize)
		self.l3 = nn.Linear(hiddenSize, outputSize)
	def forward(self, x):
		x.cuda()
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = self.l3(x)
		return x




class DefenceAgent(object):
	# hyper parameters
	EPS_START = 0.0  # e-greedy threshold start value
	EPS_END = 0.0  # e-greedy threshold end value
	EPS_DECAY = 13000  # e-greedy threshold decay
	GAMMA = 0.9  # Q-learning discount factor
	LR = 0.001  # NN optimizer learning rate
	INPUT_SIZE = 141 # input layer size
	HIDDEN_SIZE= 128  # NN hidden layer size
	OUTPUT_SIZE = 16  # output layer size
	BATCH_SIZE = 32  # Q-learning batch size
	Memory_Capacity = 50000  # memoory capacity
	UPDATE_TARGET_Q_FREQ = 300 # frequency of updating the target q
	steps_done=0
	loadWeightType = "2017ais" # for loading
	loadAgentType = "zen" # for loading
	loadWeightNum = "1250" # for loadinga3/envs/ftg/lib/python3.10/site-packages/py4j/java_gateway.py", line 1224, in send_command
	steps_learn_in_a_round=0
	model_count_in_a_round =0
	use_cuda = False
	# if gpu is to be used
	use_cuda = torch.cuda.is_available()
	FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
	LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
	ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
	Tensor = FloatTensor
	energy=torch.tensor([0.,0.,0.,0.,0.166666666666667,0.,0.5,0.,0.,0.,0.183333333333333,0.033333333333333,0.,0.,0.,0.])
	#energy=torch.tensor([0., 0., 0., 0., 0.,0., 0., 0., 0., 0.01666667,0.03333333, 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.01666667, 0.06666667, 0.,0.18333333, 0., 0.16666667, 0.01666667, 0.06666667,0.03333333, 0.13333333, 0.03333333, 0.16666667, 0.5])
	# memory
	memory = ReplayMemory(Memory_Capacity)
	# body
	model = Defence_Network(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).cuda()
	model.load_state_dict(torch.load("/home/t-yamamoto/Desktop/research/Gym-FightingICE/result/defenceai11.2best/defencedecrease1850.pth"))
	target_model = copy.deepcopy(model).cuda()
	

		
	# optimizer
	optimizer = optim.Adam(model.parameters(), LR)






	def select_action(self, state):
		sample = random.random()
		decay =  (self.EPS_START - self.EPS_END) / self.EPS_DECAY
		eps_threshold =0#self.EPS_START - (decay * self.steps_done)
		if eps_threshold<0.1:
			eps_threshold=0
		self.steps_done+=1
		if sample > eps_threshold:
			self.model_count_in_a_round += 1 
			# print("model-----------------------")
			Qvalues = self.model(Variable(state, volatile=True).type(self.FloatTensor)).data
    			
			for i in range(16):
				if state[0][1]<self.energy[i]:
					Qvalues[0][i]=-100000 
			Qvalues[0][0]=-1000000
			Qvalues[0][1]=-1000000
			Qvalues[0][11]=-1000000
			Qvalues[0][5]=-1000000 #防御AIはこれ以外で防御
			Qvalues[0][15]=-10000000
			max_totalQ_index = (Qvalues).max(1)[1].view(1,1)  
			



			# print("multi max offence Q  : {0}  multi actionName : {1}".format(max_offenceQ, self.actionMap.actionMap[int(max_offenceQ_index[0,0])]))   
			# print("multi max defence Q : {0}  multi actionName : {1}".format(max_defenceQ, self.actionMap.actionMap[int(max_defenceQ_index[0,0])]))
			# print("multi max total Q    : {0}  multi actionName : {1}".format(max_totalQ, self.actionMap.actionMap[int(max_totalQ_index[0,0])]))

			return max_totalQ_index
		else:
			# print("random-----------------------")
			x=np.array([1 for i in range(16)])
			y=np.array([])
			for i in range(16):
				if state[0][1]>=self.energy[i]:
					y=np.append(y,i)
			action=np.random.choice(y,1)
			return self.LongTensor([[action[0]]])

						

	def pushData(self,state,action, next_state, defenceReward):
		self.memory.push((self.FloatTensor([state]),action, self.FloatTensor([next_state]), self.FloatTensor([defenceReward])))

	def learn(self,episode):
		if len(self.memory) < self.BATCH_SIZE:
			return 0.0
		self.steps_learn_in_a_round += 1

		transitions = self.memory.sample(self.BATCH_SIZE)

		batch_state, batch_action, batch_next_state, batch_defence_reward = zip(*transitions)

		batch_state = Variable(torch.cat(batch_state))
		batch_action = Variable(torch.cat(batch_action))
		batch_defence_reward = Variable(torch.cat(batch_defence_reward))
		batch_next_state = Variable(torch.cat(batch_next_state))
		# current Q values are estimated by NN for all actions
		current_q_values = self.model(batch_state).gather(1, batch_action)#q
		# expected Q values are estimated from actions which gives maximum Q value
		max_next_q_values = self.target_model(batch_next_state).detach().max(1)[0]#qs
		expected_q_values = batch_defence_reward + (self.GAMMA * max_next_q_values)
		# loss is measured from error between current and newly expected Q values
		#print(current_offence_q_values.size())#[32,1]
		#print(current_offence_q_values.size())#[32]
		Loss = F.smooth_l1_loss(current_q_values.squeeze(), expected_q_values)
		# print("all_loss : {}".format(all_Loss.data[0]))
		# print("offence_loss : {}".format(offenceLoss.data[0]))
		# print("defence_loss : {}".format(defenceLoss.data[0]))


		self.optimizer.zero_grad()
		Loss.backward()
		self.optimizer.step()

		# target Q
		if self.steps_learn_in_a_round % self.UPDATE_TARGET_Q_FREQ == 0:
			self.target_model = copy.deepcopy(self.model)

		return Loss


		

class NotifyingNetwork(nn.Module):
	def __init__(self, inputSize, hiddenSize, outputSize):
		nn.Module.__init__(self)
		self.l1 = nn.Linear(inputSize,1024)
		self.l2 = nn.Linear(1024,512)
		self.l3 = nn.Linear(512, 248)




	def forward(self, x):
		x.cuda()
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = F.relu(self.l3(x))
		return x

class NotifyingattackNetwork(nn.Module):
	def __init__(self):
		nn.Module.__init__(self)
		self.l1 = nn.Linear(248,2)



	def forward(self, x):
		x.cuda()
		m = nn.LeakyReLU(0.1)
		x=self.l1(x)
		return x

	def selectforward(self, x):
		x.cuda()
		m = nn.LeakyReLU(0.1)
		x=self.l1(x)
		return x

class NotifyingdefenceNetwork(nn.Module):
	def __init__(self):
		nn.Module.__init__(self)
		self.l1 = nn.Linear(248,2)




	def forward(self, x):
		x.cuda()
		m = nn.LeakyReLU(0.1)
		x=self.l1(x)
		return x


	def selectforward(self, x):
		x.cuda()
		m = nn.LeakyReLU(0.1)
		x=self.l1(x)
		return x




class NotifyingAgent(object):
	# hyper parameters
	EPS_START = 0.0  # e-greedy threshold start value
	EPS_END = 0.0  # e-greedy threshold end value
	EPS_DECAY = 13000  # e-greedy threshold decay
	GAMMA = 0.3  # Q-learning discount factor
	LR = 0.001  # NN optimizer learning rate
	INPUT_SIZE = 157 # input layer size
	HIDDEN_SIZE= 128  # NN hidden layer size
	OUTPUT_SIZE = 16  # output layer size
	BATCH_SIZE = 32  # Q-learning batch size
	Memory_Capacity = 50000  # memoory capacity
	UPDATE_TARGET_Q_FREQ = 300 # frequency of updating the target q
	steps_done=0
	loadWeightType = "2017ais" # for loading
	loadAgentType = "zen" # for loading
	loadWeightNum = "1250" # for loading
	steps_learn_in_a_round=0
	model_count_in_a_round =0
	use_cuda = True
	# if gpu is to be used
	use_cuda = torch.cuda.is_available()
	FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
	LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
	ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
	Tensor = FloatTensor
	energy=torch.tensor([0.,0.,0.,0.,0.166666666666667,0.,0.5,0.,0.,0.,0.183333333333333,0.033333333333333,0.,0.,0.,0.])
	#energy=torch.tensor([0., 0., 0., 0., 0.,0., 0., 0., 0., 0.01666667,0.03333333, 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0., 0., 0.,0., 0., 0.01666667, 0.06666667, 0.,0.18333333, 0., 0.16666667, 0.01666667, 0.06666667,0.03333333, 0.13333333, 0.03333333, 0.16666667, 0.5])
	# memory
	memory = ReplayMemory(Memory_Capacity)
	# body
	model = NotifyingNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).cuda()
	model.load_state_dict(torch.load("/home/t-yamamoto/Desktop/research/Gym-FightingICE/researchresult/makeQ/1211(best)/notifying1999.pth"))
	target_model = copy.deepcopy(model).cuda()
	attackmodel = NotifyingattackNetwork().cuda()
	attackmodel.load_state_dict(torch.load("/home/t-yamamoto/Desktop/research/Gym-FightingICE/researchresult/makeQ/1211(best)/notifyingattack1999.pth"))
	target_attackmodel = copy.deepcopy(attackmodel).cuda()
	defencemodel = NotifyingdefenceNetwork().cuda()
	defencemodel.load_state_dict(torch.load("/home/t-yamamoto/Desktop/research/Gym-FightingICE/researchresult/makeQ/1211(best)/notifyingdefence1999.pth"))
	target_defencemodel = copy.deepcopy(defencemodel).cuda()
	

		
	# optimizer
	optimizer = optim.Adam
	def pushData(self,state,action, next_state, defenceReward,attackReward):
		self.memory.push((self.FloatTensor([state]),action, self.FloatTensor([next_state]), self.FloatTensor([defenceReward]), self.FloatTensor([attackReward])))

	def learn(self,episode):
		if len(self.memory) < self.BATCH_SIZE:
			return 0.0,0.0,0.0
		self.steps_learn_in_a_round += 1

		transitions = self.memory.sample(self.BATCH_SIZE)

		batch_state, batch_action, batch_next_state, batch_defence_reward,batch_attack_reward = zip(*transitions)

		batch_state = Variable(torch.cat(batch_state))
		batch_action = Variable(torch.cat(batch_action))
		batch_defence_reward = Variable(torch.cat(batch_defence_reward))
		batch_attack_reward = Variable(torch.cat(batch_attack_reward))
		batch_next_state = Variable(torch.cat(batch_next_state))


		current_q_values = self.model(batch_state)

		#Defence
		current_defenceq_values=self.defencemodel(current_q_values).gather(1, batch_action)#q
		# expected Q values are estimated from actions which gives maximum Q value
		max_next_defenceq_values = self.target_defencemodel(self.target_model(batch_next_state)).detach().max(1)[0]#qs
		expected_defenceq_values = batch_defence_reward + (self.GAMMA * max_next_defenceq_values)
		# loss is measured from error between current and newly expected Q values
		defenceLoss = F.smooth_l1_loss(current_defenceq_values.squeeze(), expected_defenceq_values)

		#attack
		current_attackq_values=self.attackmodel(current_q_values).gather(1, batch_action)#q
		# expected Q values are estimated from actions which gives maximum Q value
		max_next_attackq_values = self.target_attackmodel(self.target_model(batch_next_state)).detach().max(1)[0]#qs
		expected_attackq_values = batch_attack_reward + (self.GAMMA * max_next_attackq_values)
		# loss is measured from error between current and newly expected Q values
		attackLoss = F.smooth_l1_loss(current_attackq_values.squeeze(), expected_attackq_values)

		Loss=defenceLoss+attackLoss

		self.optimizer.zero_grad()
		self.defenceoptimizer.zero_grad()
		self.attackoptimizer.zero_grad()
		Loss.backward()
		self.optimizer.step()
		self.defenceoptimizer.step()
		self.attackoptimizer.step()
		torch.save(self.target_model.state_dict(),'/home/t-yamamoto/Desktop/research/Gym-FightingICE/model4/notifying'+str(episode)+'.pth')
		torch.save(self.target_defencemodel.state_dict(),'/home/t-yamamoto/Desktop/research/Gym-FightingICE/model5/notifyingdefence'+str(episode)+'.pth')
		torch.save(self.target_attackmodel.state_dict(),'/home/t-yamamoto/Desktop/research/Gym-FightingICE/model6/notifyingattack'+str(episode)+'.pth')

		# target Q
		if self.steps_learn_in_a_round % self.UPDATE_TARGET_Q_FREQ == 0:
			self.target_model = copy.deepcopy(self.model)
			self.defencetarget_model = copy.deepcopy(self.defencemodel)
			self.attacktarget_model = copy.deepcopy(self.attackmodel)

		return Loss,attackLoss,defenceLoss(model.parameters(), LR)
	attackoptimizer = optim.Adam(attackmodel.parameters(), LR)
	defenceoptimizer = optim.Adam(defencemodel.parameters(), LR)





	def select_action(self, state,switch):
		sample = random.random()
		decay =  (self.EPS_START - self.EPS_END) / self.EPS_DECAY
		eps_threshold =self.EPS_START - (decay * self.steps_done)
		if eps_threshold<0.1:
			eps_threshold=0.01
		self.steps_done+=1
		if sample > eps_threshold:
			self.model_count_in_a_round += 1 
			# print("model-----------------------")
			attackQ= self.attackmodel.selectforward(self.model(Variable(state, volatile=True)).type(self.FloatTensor)).data
			defenceQ=self.defencemodel.selectforward(self.model(Variable(state, volatile=True)).type(self.FloatTensor)).data
			for i in range(attackQ.size(0)):
				attackQ[i][1]=0	
			print("------------------------")
			#print("attackQ",end="")
			#print(attackQ)
			print("defenceQ",end="")
			print(defenceQ)
			attacksoft=F.softmax(attackQ/0.2,dim=1)
			defencesoft=F.softmax(defenceQ/0.2,dim=1)
			#print("attacksoft",end="")
			#print(attacksoft)
			print("defencesoft",end="")
			print(defencesoft)
			Qvalues =attacksoft+1*defencesoft #0.9yosage(0.7のほうがよさげxyk)
			#print("addQ",end="")
			print("before",end="")
			print(Qvalues)
			#Qvalues=F.softmax(Qvalues/0.3,dim=1)#/0.3してもよさげ
			print(Qvalues)
			print("------------------------")
			return (Qvalues).max(1)[1].view(1,1) 
			#x=nn.Softmax(dim=1)
			#Qvalues=x(Qvalues)
			
			#print(Qvalues[0])
			#print(Qvalues)
			#print(Qvalues.size())

		else:
			action=np.random.randint(0,2)
			return self.LongTensor([[action]])

						

	def pushData(self,state,action, next_state, defenceReward,attackReward):
		self.memory.push((self.FloatTensor([state]),action, self.FloatTensor([next_state]), self.FloatTensor([defenceReward]), self.FloatTensor([attackReward])))

	def learn(self,episode):
		if len(self.memory) < self.BATCH_SIZE:
			return 0.0,0.0,0.0
		self.steps_learn_in_a_round += 1

		transitions = self.memory.sample(self.BATCH_SIZE)

		batch_state, batch_action, batch_next_state, batch_defence_reward,batch_attack_reward = zip(*transitions)

		batch_state = Variable(torch.cat(batch_state))
		batch_action = Variable(torch.cat(batch_action))
		batch_defence_reward = Variable(torch.cat(batch_defence_reward))
		batch_attack_reward = Variable(torch.cat(batch_attack_reward))
		batch_next_state = Variable(torch.cat(batch_next_state))


		current_q_values = self.model(batch_state)

		#Defence
		current_defenceq_values=self.defencemodel(current_q_values).gather(1, batch_action)#q
		# expected Q values are estimated from actions which gives maximum Q value
		max_next_defenceq_values = self.target_defencemodel(self.target_model(batch_next_state)).detach().max(1)[0]#qs
		expected_defenceq_values = batch_defence_reward + (self.GAMMA * max_next_defenceq_values)
		# loss is measured from error between current and newly expected Q values
		defenceLoss = F.smooth_l1_loss(current_defenceq_values.squeeze(), expected_defenceq_values)

		#attack
		current_attackq_values=self.attackmodel(current_q_values).gather(1, batch_action)#q
		# expected Q values are estimated from actions which gives maximum Q value
		max_next_attackq_values = self.target_attackmodel(self.target_model(batch_next_state)).detach().max(1)[0]#qs
		expected_attackq_values = batch_attack_reward + (self.GAMMA * max_next_attackq_values)
		# loss is measured from error between current and newly expected Q values
		attackLoss = F.smooth_l1_loss(current_attackq_values.squeeze(), expected_attackq_values)

		Loss=defenceLoss+attackLoss

		self.optimizer.zero_grad()
		self.defenceoptimizer.zero_grad()
		self.attackoptimizer.zero_grad()
		Loss.backward()
		self.optimizer.step()
		self.defenceoptimizer.step()
		self.attackoptimizer.step()
		torch.save(self.target_model.state_dict(),'/home/t-yamamoto/Desktop/research/Gym-FightingICE/model4/notifying'+str(episode)+'.pth')
		torch.save(self.target_defencemodel.state_dict(),'/home/t-yamamoto/Desktop/research/Gym-FightingICE/model5/notifyingdefence'+str(episode)+'.pth')
		torch.save(self.target_attackmodel.state_dict(),'/home/t-yamamoto/Desktop/research/Gym-FightingICE/model6/notifyingattack'+str(episode)+'.pth')

		# target Q
		if self.steps_learn_in_a_round % self.UPDATE_TARGET_Q_FREQ == 0:
			self.target_model = copy.deepcopy(self.model)
			self.defencetarget_model = copy.deepcopy(self.defencemodel)
			self.attacktarget_model = copy.deepcopy(self.attackmodel)

		return Loss,attackLoss,defenceLoss




def calc_distance(x1,y1,x2,y2):
	return ((x1-x2)**2+(y1-y2)**2)**0.5


						


def opp_state(state):
	for i in range(56):
		if state[73+i]==1:
			return i

def rtod(x):
	y=np.zeros(2)
	if x>0:
		y[1]=1.0
	else:
		y[0]=1.0
	return y
"入力で入れたやつ"
from pynput import keyboard
from pynput.keyboard import Key, Listener,KeyCode
from timeout_decorator import timeout, TimeoutError


TIMEOUT_SEC = 0.3
@timeout(TIMEOUT_SEC)
def inputaction():
    with keyboard.Events() as events:
        for event in events:
            print("key")
            print(event.key)
            if event.key==KeyCode.from_char('a'):
                action=0#"STAND_A"
            elif event.key==KeyCode.from_char('*'):
                action=1#JUMP
            elif event.key==keyboard.Key.backspace:
                action=2#BACK_STEP
            elif event.key==KeyCode.from_char('f'):
                action=3#FORWARD_WALK
            elif event.key==keyboard.Key.down:
                action=4#"STAND_D_DB_BB"
            elif event.key==KeyCode.from_char('b'):
                action=5# "STAND_B"
            elif event.key==KeyCode.from_char('c'):
                action=6#STAND_D_DF_FC"
            elif event.key==KeyCode.from_char('x'):
                action=7# "STAND_F_D_DFA"
            elif event.key==KeyCode.from_char('y'):
                action=8#"STAND_FB"
            elif event.key==KeyCode.from_char('g'):
                action=9#"CROUCH_GUARD"
            elif event.key==KeyCode.from_char('q'):
                action=10#"STAND_F_D_DFB"
            elif event.key==KeyCode.from_char('s'):
                action=11#"THROW_B"
            elif event.key==KeyCode.from_char('k'):
                action=12#"CROUCH_FB"
            elif event.key==KeyCode.from_char('v'):
                action=13#"BACKJUMP
            elif event.key==KeyCode.from_char('w'):
                action=14#FORJUMP
            else:
                action=15#NEUTRAL
            return action
       
@timeout(TIMEOUT_SEC)
def reverseinputaction():
    with keyboard.Events() as events:
        for event in events:
            print("key")
            print(event.key)
            if event.key==KeyCode.from_char('a'):
                action=0#"STAND_A"
            elif event.key==KeyCode.from_char('*'):
                action=1#FOR_JUMP
            elif event.key==keyboard.Key.backspace:
                action=3#FORWARD_WALK
            elif event.key==KeyCode.from_char('f'):
                action=2#BACK_STEP
            elif event.key==keyboard.Key.down:
                action=4#"STAND_D_DB_BB"
            elif event.key==KeyCode.from_char('b'):
                action=5# "STAND_B"
            elif event.key==KeyCode.from_char('c'):
                action=6#STAND_D_DF_FC"
            elif event.key==KeyCode.from_char('x'):
                action=7# "STAND_F_D_DFA"
            elif event.key==KeyCode.from_char('y'):
                action=8#"STAND_FB"
            elif event.key==KeyCode.from_char('g'):
                action=9#"CROUCH_GUARD"
            elif event.key==KeyCode.from_char('q'):
                action=10#"STAND_F_D_DFB"
            elif event.key==KeyCode.from_char('s'):
                action=11#"THROW_B"
            elif event.key==KeyCode.from_char('k'):
                action=12#"CROUCH_FB"
            elif event.key==KeyCode.from_char('v'):
                action=14#"CROUCH_FB"
            elif event.key==KeyCode.from_char('w'):
                action=13#"CROUCH_FB"
            else:
                action=15#yyNEUTRAL
            return action





def main():
	sys.path.append('gym-fightingice')
	episodes=10
	sync_interval=300
	defenceagent=DefenceAgent()
	notifyingagent=NotifyingAgent()
	total_reward=0
	total_win=0
	win_rate=0.0
	switch=0
	timestep=0
	for episode in range(episodes):
		if episode==0:
			env = gym.make("FightingiceDataFrameskip-v0", java_env_path="", port=1234)
		nostate=env.reset(p2=Machete)	
		state=nostate[0][0]
		done=False
		epireward=0.0
		loss=0.0
		before_action=100
		ableaction=True
		roundtimestep=0
		numaction=0
		numnotify=0
		while not done:
			try:
				if state[2]<state[66]:
					action=inputaction()
				else:
					action=reverseinputaction()
			except TimeoutError:
					action=15
			if action==6:
				if state[1]<0.5:
					action=15
			elif action==10:
				if state[1]<0.18333333333:
					action=15
			notifyingstate=state
			for i in range(16):
				if i==action:
					notifyingstate=np.append(notifyingstate,1.)
				else:
					notifyingstate=np.append(notifyingstate,0.)
			if state[0]>state[64]:
				switch=1
			else:
				switch=0
			notify=notifyingagent.select_action(defenceagent.FloatTensor([notifyingstate]),switch)
			print(notify)
			if notify==1:
				numnotify+=1
				action = defenceagent.select_action(defenceagent.FloatTensor([state]))
			else:
				action=torch.LongTensor([[action]])
			#wandb.log({'timestep':timestep, 'notify':notify})
			#wandb.log({'timestep':timestep, 'action':action})
			numaction+=1
			print("action")
			print(action)
			next_state,ableaction,done,info=env.step(int(action)) #2timestep(0.033s)		
			timestep+=1
			roundtimestep+=1
			innext_state=next_state[0]
			action=random.randint(0,15)
			notifyingnext_state=innext_state
			for i in range(16):
				if i==action:
					notifyingnext_state=np.append(notifyingnext_state,1.)
				else:
					notifyingnext_state=np.append(notifyingnext_state,0.)
			state =innext_state
			notifyingstate=notifyingnext_state
			#notifying_loss,attack_loss,defence_loss=notifyingagent.learn(episode)
			#wandb.log({'timestep':timestep, 'loss':notifying_loss})
			#wandb.log({'timestep':timestep, 'attack_loss':attack_loss})
			#wandb.log({'timestep':timestep, 'defence_loss':defence_loss})
		if innext_state[0]>innext_state[64]:
			total_win+=1
		score=1000*innext_state[0]/float(innext_state[0]+innext_state[64])
		wandb.log({'episode': episode, 'total_win': total_win})#いままで得た報酬
		#wandb.log({'episode': episode, 'epireward': epireward})#いままで得た報酬
		wandb.log({'episode': episode, 'my_HP':400*innext_state[0]})#そのエピソードの報酬
		wandb.log({'episode': episode, 'opp_HP':400*innext_state[64]})#そのエピソードの報酬
		wandb.log({'episode': episode, 'score':score})#そのエピソードの報酬
		wandb.log({'episode': episode, 'control dash rate':numnotify/float(numaction)})#そのエピソードの報酬



if __name__ == "__main__":
	main()
