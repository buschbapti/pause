import numpy as np
from collections import deque
import json

class Assessment(object):
    def __init__(self,window_size):
        self.window_size = window_size
        self.window_score = deque(maxlen=window_size)
        # initialize the reba table
        self.reba_table_init()

    def reba_table_init(self):
        # group A table
        self.reba_A_table = np.zeros((3,5,4))
        self.reba_A_table[0] = [[1,2,3,4],[2,3,4,5],[2,4,5,6],[3,5,6,7],[4,6,7,8]]
        self.reba_A_table[1] = [[1,2,3,4],[3,4,5,6],[4,5,6,7],[5,6,7,8],[6,7,8,9]]
        self.reba_A_table[2] = [[3,3,5,6],[4,5,6,7],[5,6,7,8],[6,7,8,9],[7,7,8,9]]
        # group B table
        self.reba_B_table = np.zeros((2,6,3))
        self.reba_B_table[0] = [[1,2,2],[1,2,3],[3,4,5],[4,5,5],[6,7,8],[7,8,8]]
        self.reba_B_table[1] = [[1,2,3],[2,3,4],[4,5,5],[5,6,7],[7,8,8],[8,9,9]]
        # joint C table
        self.reba_C_table = np.zeros((12,12))
        self.reba_C_table[0] = [1,1,1,2,3,3,4,5,6,7,7,7]
        self.reba_C_table[1] = [1,2,2,3,4,4,5,6,6,7,7,8]
        self.reba_C_table[2] = [2,3,3,3,4,5,6,7,7,8,8,8]
        self.reba_C_table[3] = [3,4,4,4,5,6,7,8,8,9,9,9]
        self.reba_C_table[4] = [4,4,4,5,6,7,8,8,9,9,9,9]
        self.reba_C_table[5] = [6,6,6,7,8,8,9,9,10,10,10,10]
        self.reba_C_table[6] = [7,7,7,8,9,9,9,10,10,11,11,11]
        self.reba_C_table[7] = [8,8,8,9,10,10,10,10,10,11,11,11]
        self.reba_C_table[8] = [9,9,9,10,10,10,11,11,11,12,12,12]
        self.reba_C_table[9] = [10,10,10,11,11,11,11,12,12,12,12,12]
        self.reba_C_table[10] = [11,11,11,11,12,12,12,12,12,12,12,12]
        self.reba_C_table[11] = [12,12,12,12,12,12,12,12,12,12,12,12]

    def assess(self,data,method='reba',epsilon=5):
        if method == 'reba':
            score = self.reba_assess(data,epsilon)
        self.window_score.append(score)
        return score

    def eval_score(self):
        return np.sum(self.window_score)/len(self.window_score)

    def save_log(self, score, group):
        # read the json file score
        with open('/tmp/pause_log.json') as data_file:
            log_data = json.load(data_file)
        # append the score
        log_data[group] = score
        with open('/tmp/pause_log.json', 'w') as outfile:
            json.dump(log_data, outfile, indent=4, sort_keys=True)

    def reba_assess(self,data,epsilon=5):
        # group A score calculation (trunk,neck,legs)
        def get_A_score():
            trunk_score = 0
            neck_score = 0
            legs_score = 0
            ########### TRUNK ###############
            # flexion/extension
            if abs(data["trunk"][0]) < epsilon:
                trunk_score += 1
            elif abs(data["trunk"][0]) < 20 + epsilon:
                trunk_score += 2
            elif abs(data["trunk"][0]) < 60 + epsilon:
                trunk_score += 3
            else:
                trunk_score += 4
            # side bending/twisting
            if abs(data["trunk"][1]) > epsilon or abs(data["trunk"][2]) > epsilon:
                trunk_score += 1
            ########## NECK ############
            # flexion/extension
            if data["neck"][0] < epsilon and data["neck"][0] > -20 - epsilon:
                neck_score += 1
            else:
                neck_score += 2
            # side bending/twisting
            if abs(data["neck"][1]) > epsilon or abs(data["neck"][2]) > epsilon:
                neck_score += 1
            ########## LEGS ###########
            # both legs in contact with the ground
            if data["leg_R"][1] == 1 and data["leg_L"][1] == 1:
                legs_score += 1
            else:
                legs_score += 2
            # knees inclination
            legs = np.maximum(np.absolute(data["leg_R"]),np.absolute(data["leg_L"]))
            if legs[0] > 60 - epsilon:
                legs_score += 2
            elif legs[0] > 30 - epsilon:
                legs_score += 1
            ######### A_SCORE ########
            score = self.reba_A_table[neck_score-1,trunk_score-1,legs_score-1] 
            ######### LOAD ###########
            if data["load"] > 5 and data["load"] < 10:
                score += 1
            elif data["load"] > 10:
                score += 2
            # write the log
            log_score = {}
            log_score["trunk"] = trunk_score
            log_score["neck"] = neck_score
            log_score["legs"] = legs_score
            log_score["A_score"] = score
            self.save_log(log_score,'groupA')
            # return the score
            return score
        # group B score calculation (shoulders,elbows,wrists)
        def get_B_score():
            shoulders_score = 0
            elbows_score = 0
            wrists_score = 0
            ######### SHOULDERS ##########
            # flexion/extension
            shoulder = np.maximum(np.absolute(data["shoulder_R"]),np.absolute(data["shoulder_L"]))
            elbows = np.maximum(np.absolute(data["elbow_R"]),np.absolute(data["elbow_L"]))
            if shoulder[0] < 20 + epsilon:
                shoulders_score += 1
            elif shoulder[0] < 45 + epsilon:
                shoulders_score += 2
            elif shoulder[0] < 90 + epsilon:
                shoulders_score += 3
            else:
                shoulders_score += 4
            # abduction/rotation
            if shoulder[1] > epsilon or shoulder[2] > epsilon or elbows[1] > epsilon:
                shoulders_score += 1
            ########## ELBOWS ############
            # flexion/etension
            if elbows[0] > 60 - epsilon and elbows[0] < 100 + epsilon:
                elbows_score += 1
            else:
                elbows_score += 2
            ######### WRISTS ###########
            # flexion/extension
            wrists = np.maximum(np.absolute(data["wrist_R"]),np.absolute(data["wrist_L"]))
            if wrists[0] < 15 + epsilon:
                wrists_score += 1
            else:
                wrists_score += 2
            # deviation/twisting
            if wrists[1] > epsilon or wrists[2] > epsilon:
                wrists_score += 1
            # return the score
            score = self.reba_B_table[elbows_score-1,shoulders_score-1,wrists_score-1]
            # write the log
            log_score = {}
            log_score["shoulders"] = shoulders_score
            log_score["elbows"] = elbows_score
            log_score["wrists"] = wrists_score
            log_score["B_score"] = score
            self.save_log(log_score,'groupB')
            # return the score
            return score
        # first get A and B score
        A_score = get_A_score()
        B_score = get_B_score()
        # calculate reba score from reba table
        reba_score = self.reba_C_table[A_score-1,B_score-1]
        # add activity score
        if data["static"] or data["high_dynamic"]:
            reba_score += 1
        # save the log
        self.save_log(reba_score,"REBA")
        # return reba score
        return reba_score



