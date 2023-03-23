import numpy as np
from py4j.java_gateway import get_field
import time


		
		
class GymAI(object):
    def __init__(self, gateway, pipe, frameskip=True):
        self.gateway = gateway
        self.pipe = pipe

        self.width = 96  # The width of the display to obtain
        self.height = 64  # The height of the display to obtain
        self.grayscale = True  # The display's color to obtain true for grayscale, false for RGB

        self.obs = None
        self.just_inited = True

        self._actions = "AIR_A AIR_B AIR_D_DB_BA AIR_D_DB_BB AIR_D_DF_FA AIR_D_DF_FB AIR_DA AIR_DB AIR_F_D_DFA AIR_F_D_DFB AIR_FA AIR_FB AIR_GUARD AIR_UA AIR_UB BACK_JUMP BACK_STEP  CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB CROUCH_GUARD FOR_JUMP FORWARD_WALK JUMP STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FA STAND_D_DF_FB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB STAND_GUARD THROW_A THROW_B"
        self.action_strs = self._actions.split(" ")

        self.pre_framedata = None

        self.frameskip = frameskip




    def close(self):
        pass

    def initialize(self, gameData, player):
        self.inputKey = self.gateway.jvm.struct.Key()
        self.frameData = self.gateway.jvm.struct.FrameData()
        self.cc = self.gateway.jvm.aiinterface.CommandCenter()
        self.player = player 


        self.gameData = gameData
        self.action=0
        self.ableaction=True
        self.start=self.gateway.jvm.struct.FrameData()
        self.ret_pre_framedata=self.start
        self.oppstate=100
        return 0

    # please define this method when you use FightingICE version 3.20 or later
    def roundEnd(self, x, y, z):
        print("send round end to {}".format(self.pipe))
        self.pipe.send([self.obs, 0, True, None])
        self.just_inited = True
        # request = self.pipe.recv()
        # if request == "close":
        #     return
        self.obs = None
        self.ret_pre_framedata=self.start

    # Please define this method when you use FightingICE version 4.00 or later
    def getScreenData(self, sd):
        self.screenData = sd

    def getInformation(self, frameData, isControl):
        self.pre_framedata = frameData if self.pre_framedata is None else self.frameData
        self.frameData = frameData
        self.isControl = isControl
        self.cc.setFrameData(self.frameData, self.player)
        if frameData.getEmptyFlag():
            return

    def input(self):
        return self.inputKey

    def gameEnd(self):
        pass
  

    def processing(self):
        
        #time.sleep(0.001)
        #print(self.ableaction)
        if self.frameData.getEmptyFlag() or self.frameData.getRemainingTime() <= 0:
            self.isGameJustStarted = True
            return
        if self.frameskip:
            if self.cc.getSkillFlag():
                self.inputKey = self.cc.getSkillKey()
                return
            if not self.isControl:
                return



        # if just inited, should wait for first reset()
        if self.just_inited:
            request = self.pipe.recv()
            if request == "reset":
                self.just_inited = False
                self.obs = self.get_obs()
                self.pipe.send(self.obs)
            else:
                raise ValueError
        # if not just inited but self.obs is none, it means second/thrid round just started
        # should return only obs for reset()
        elif self.obs is None:  

            self.obs = self.get_obs()
            self.pipe.send(self.obs)
        # if there is self.obs, do step() and return [obs, reward, done, info]
        else:
            self.obs = self.get_obs()
            self.reward = self.get_reward()
            self.ret_pre_framedata=self.frameData
            self.pipe.send([self.obs, self.reward, False, None])

        #print("waitting for step in {}".format(self.pipe))
        request = self.pipe.recv()
        #print("get step in {}".format(self.pipe))
        if len(request) == 2 and request[0] == "step":
            self.action = request[1]
            if self.action!=9:
               self.inputKey.empty()
            self.cc.skillCancel()
            trans_action_name=self.action_strs[self.action]
            if isinstance(self.action,int):
                self.cc.commandCall(trans_action_name)
            else:
                self.cc.commandCall(trans_action_name)
            if not self.frameskip:
                self.inputKey = self.cc.getSkillKey()

    def get_reward(self):
       
        try:
            if self.pre_framedata.getEmptyFlag() or self.frameData.getEmptyFlag():
                reward = 0 


            else:
                #p2_hp_pre = self.pre_framedata.getCharacter(False).getHp()
                p1_hp_pre = self.ret_pre_framedata.getCharacter(True).getHp()
                #print("p1 pre"+str(p1_hp_pre))
                #p2_hp_now = self.frameData.getCharacter(False).getHp()
                p1_hp_now = self.frameData.getCharacter(True).getHp()
                #print("p1"+str(p1_hp_now))


                if self.player:
                    reward = (p1_hp_pre-p1_hp_now)#+7.0*(p1_energy_pre-p1_energy_now)
                else:
                    reward = (p1_hp_pre-p1_hp_now) #- (p2_hp_pre-p2_hp_now)
        except:
            reward = 0
        return reward

    def get_obs(self):


        my = self.frameData.getCharacter(self.player)
        opp = self.frameData.getCharacter(not self.player)
        combmy = self.frameData.getCharacter(True)
        p1comb = combmy.getHitCount()
        #print("p1conb="+str(p1comb))
        # my information
        myHp = abs(my.getHp() / 10000.0)
        myEnergy = my.getEnergy() / 300
        myX = ((my.getLeft() + my.getRight()) / 2) / 960
        #print(myX,end="")
        myY = ((my.getBottom() + my.getTop()) / 2) / 640
        mySpeedX = my.getSpeedX() / 15
        mySpeedY = my.getSpeedY() / 28
        myState = my.getAction().ordinal()
        myRemainingFrame = my.getRemainingFrame() / 70
        mycomb = my.getHitCount()
        # opp information
        oppHp = abs(opp.getHp() / 10000.0)
        oppEnergy = opp.getEnergy() / 300
        oppX = ((opp.getLeft() + opp.getRight()) / 2) / 960
        oppY = ((opp.getBottom() + opp.getTop()) / 2) / 640
        oppSpeedX = opp.getSpeedX() / 15
        oppSpeedY = opp.getSpeedY() / 28
        oppState = opp.getAction().ordinal()
        oppRemainingFrame = opp.getRemainingFrame() / 70

        # time information
        game_frame_num = self.frameData.getFramesNumber() / 3600
        #print("frame:"+str(game_frame_num*3600) )
        observation = []

        # my information
        observation.append(myHp)#0
        observation.append(myEnergy)#1
        observation.append(myX)#2
        observation.append(myY)#3
        if mySpeedX < 0:
            observation.append(0)#4
        else:
            observation.append(1)
        observation.append(abs(mySpeedX))#5
        if mySpeedY < 0:
            observation.append(0)#6
        else:
            observation.append(1)
        observation.append(abs(mySpeedY))#7
        for i in range(56):#8-63
            if i == myState:
                observation.append(1)
            else:
                observation.append(0)
        #observation.append(myRemainingFrame)

        # opp information
        observation.append(oppHp)#64
        observation.append(oppEnergy)#65
        observation.append(oppX)#66
        observation.append(oppY)#67
        if oppSpeedX < 0:#68
            observation.append(0)
        else:
            observation.append(1)
        observation.append(abs(oppSpeedX))#69
        if oppSpeedY < 0:#70
            observation.append(0)
        else:
            observation.append(1)
        observation.append(abs(oppSpeedY))#71
        for i in range(56):#127
            if i == oppState:
                observation.append(1)
            else:
                observation.append(0)
        #observation.append(oppRemainingFrame)

        # time information
        observation.append(game_frame_num)#127

        myProjectiles = self.frameData.getProjectilesByP1()
        oppProjectiles = self.frameData.getProjectilesByP2()

        if len(myProjectiles) == 2:
            myHitDamage = myProjectiles[0].getHitDamage() / 200.0
            myHitAreaNowX = ((myProjectiles[0].getCurrentHitArea().getLeft() + myProjectiles[
                0].getCurrentHitArea().getRight()) / 2) / 960.0
            myHitAreaNowY = ((myProjectiles[0].getCurrentHitArea().getTop() + myProjectiles[
                0].getCurrentHitArea().getBottom()) / 2) / 640.0
            observation.append(myHitDamage)#128
            observation.append(myHitAreaNowX)#129
            observation.append(myHitAreaNowY)#130
            myHitDamage = myProjectiles[1].getHitDamage() / 200.0
            myHitAreaNowX = ((myProjectiles[1].getCurrentHitArea().getLeft() + myProjectiles[
                1].getCurrentHitArea().getRight()) / 2) / 960.0
            myHitAreaNowY = ((myProjectiles[1].getCurrentHitArea().getTop() + myProjectiles[
                1].getCurrentHitArea().getBottom()) / 2) / 640.0
            observation.append(myHitDamage)#131
            observation.append(myHitAreaNowX)#132
            observation.append(myHitAreaNowY)#133
        elif len(myProjectiles) == 1:
            myHitDamage = myProjectiles[0].getHitDamage() / 200.0
            myHitAreaNowX = ((myProjectiles[0].getCurrentHitArea().getLeft() + myProjectiles[
                0].getCurrentHitArea().getRight()) / 2) / 960.0
            myHitAreaNowY = ((myProjectiles[0].getCurrentHitArea().getTop() + myProjectiles[
                0].getCurrentHitArea().getBottom()) / 2) / 640.0
            observation.append(myHitDamage)
            observation.append(myHitAreaNowX)
            observation.append(myHitAreaNowY)
            for t in range(3):
                observation.append(0.0)
        else:
            for t in range(6):
                observation.append(0.0)

        if len(oppProjectiles) == 2:
            oppHitDamage = oppProjectiles[0].getHitDamage() / 200.0
            oppHitAreaNowX = ((oppProjectiles[0].getCurrentHitArea().getLeft() + oppProjectiles[
                0].getCurrentHitArea().getRight()) / 2) / 960.0
            oppHitAreaNowY = ((oppProjectiles[0].getCurrentHitArea().getTop() + oppProjectiles[
                0].getCurrentHitArea().getBottom()) / 2) / 640.0
            observation.append(oppHitDamage)
            observation.append(oppHitAreaNowX)
            observation.append(oppHitAreaNowY)
            oppHitDamage = oppProjectiles[1].getHitDamage() / 200.0
            oppHitAreaNowX = ((oppProjectiles[1].getCurrentHitArea().getLeft() + oppProjectiles[
                1].getCurrentHitArea().getRight()) / 2) / 960.0
            oppHitAreaNowY = ((oppProjectiles[1].getCurrentHitArea().getTop() + oppProjectiles[
                1].getCurrentHitArea().getBottom()) / 2) / 640.0
            observation.append(oppHitDamage)
            observation.append(oppHitAreaNowX)
            observation.append(oppHitAreaNowY)
        elif len(oppProjectiles) == 1:
            oppHitDamage = oppProjectiles[0].getHitDamage() / 200.0
            oppHitAreaNowX = ((oppProjectiles[0].getCurrentHitArea().getLeft() + oppProjectiles[
                0].getCurrentHitArea().getRight()) / 2) / 960.0
            oppHitAreaNowY = ((oppProjectiles[0].getCurrentHitArea().getTop() + oppProjectiles[
                0].getCurrentHitArea().getBottom()) / 2) / 640.0
            observation.append(oppHitDamage)
            observation.append(oppHitAreaNowX)
            observation.append(oppHitAreaNowY)  


            for t in range(3):
                observation.append(0.0)
        else:
            for t in range(6):
                observation.append(0.0)#140

        observation = np.array(observation, dtype=np.float32)
        observation = np.clip(observation, 0, 1)
        return observation,0,0

    # This part is mandatory
    class Java:
        implements = ["aiinterface.AIInterface"]
