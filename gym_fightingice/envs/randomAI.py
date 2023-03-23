from py4j.java_gateway import get_field
import random
ac=[		  "STAND_A",
		  "JUMP",
		  "BACK_STEP",
		  "FORWARD_WALK",
		
		  "STAND_D_DB_BB",
		  "STAND_B",
		
		  "STAND_D_DF_FC",
		
		  "STAND_F_D_DFA",
		
		  "STAND_FB",
		
		 "CROUCH_GUARD",
		  "STAND_F_D_DFB",
		
		 "THROW_B",
		 "CROUCH_FB",
		 "BACK_JUMP",
		 "FOR_JUMP",		
		 "NEUTRAL"]
"""
ac=['NEUTRAL','FORWARD_WALK','DASH','BACK_STEP','JUMP','FOR_JUMP','BACK_JUMP','STAND_GUARD',
'CROUCH_GUARD',
'AIR_GUARD',
'THROW_A',
'THROW_B',
'STAND_A',
'STAND_B',
'CROUCH_A',
'CROUCH_B',
'AIR_A',
'AIR_B',
'AIR_DA',
'AIR_DB',
'STAND_FA',
'STAND_FB',
'CROUCH_FA',
'CROUCH_FB',
'AIR_FA',
'AIR_FB',
'AIR_UA',
'AIR_UB',
'STAND_D_DF_FA',
'STAND_D_DF_FB',
'STAND_F_D_DFA',
'STAND_F_D_DFB',
'STAND_D_DB_BA',
'STAND_D_DB_BB',
'AIR_D_DF_FA',
'AIR_D_DF_FB',
'AIR_F_D_DFA',
'AIR_F_D_DFB',
'AIR_D_DB_BA',
'AIR_D_DB_BB',
'STAND_D_DF_FC']
"""
class randomAI(object):
    def __init__(self, gateway):
        self.gateway = gateway
        
    def close(self):
        pass
        
    def getInformation(self, frameData, isControl):
        # Getting the frame data of the current frame
        self.frameData = frameData
        self.cc.setFrameData(self.frameData, self.player)
    # please define this method when you use FightingICE version 3.20 or later
    def roundEnd(self, x, y, z):
    	print(x)
    	print(y)
    	print(z)
    	
    # please define this method when you use FightingICE version 4.00 or later
    def getScreenData(self, sd):
    	pass
        
    def initialize(self, gameData, player):
        # Initializng the command center, the simulator and some other things
        self.inputKey = self.gateway.jvm.struct.Key()
        self.frameData = self.gateway.jvm.struct.FrameData()
        self.cc = self.gateway.jvm.aiinterface.CommandCenter()
        self.stop=0
        self.player = player
        self.gameData = gameData
        self.simulator = self.gameData.getSimulator()
                
        return 0
        
    def input(self):
        # Return the input for the current frame
        return self.inputKey
        
    def processing(self):
        # Just compute the input for the current frame
        if self.frameData.getEmptyFlag() or self.frameData.getRemainingFramesNumber() <= 0:
                self.isGameJustStarted = True
                return
        distance = self.frameData.getDistanceX()      
        if self.cc.getSkillFlag():
                self.inputKey = self.cc.getSkillKey()
                return
            
        self.inputKey.empty()
        self.cc.skillCancel()  
        if self.stop%2==0:   
           if distance > 150:
              # If its too far, then jump to get closer fast
              self.cc.commandCall("6 6")
           else:
              action=random.randint(1,15)
              self.stop+=1 
              self.cc.commandCall(ac[action]) 
        else: 
           action=15
           self.stop+=1          
           self.cc.commandCall(ac[action])
                        
    # This part is mandatory
    class Java:
        implements = ["aiinterface.AIInterface"]
        
