#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 21:58:51 2018
hw1

@author:Huitong Pan, hp4zw
"""
import numpy as np
import random

# ==================== Question 7 ======================
def generate_episode():
    states=['c1','c2','c3','pass','pub','fb','sleep']
    
    #transitional possibilities
    #  'c1','c2','c3','pass','pub','fb','sleep'
    P=[[0  ,0.5,0    ,0     ,0    ,0.5 ,0],
       [0  ,0  ,0.8  ,0     ,0    ,0   ,0.2],
       [0  ,0  ,0    ,0.6   ,0.4  ,0   ,0],
       [0  ,0  ,0    ,0     ,0    ,0   ,1],
       [0.2,0.4,0.4  ,0     ,0    ,0   ,0],
       [0.1,0  ,0    ,0     ,0    ,0.9 ,0],
       [0  ,0  ,0    ,0     ,0    ,0   ,1]]
    R=[-2,-2,-2,10,1,-1,0]
    
    myepisode='c1'
    s=0 #initial state is 0: c1
    r='-2' #initial reward is c1
    while s<6:
        currentstate=P[s]
        x=random.uniform(0, 1)
        cumulative_prob=0
        for i in range(0,7):
            cumulative_prob+=currentstate[i]
            if x<cumulative_prob:
                s=i
                myepisode=myepisode+' '+states[i]
                r=r+' '+str(R[s])
                break
    print("Episode:(states)")
    print(myepisode)
    print("Rewards:")
    print(r)        

generate_episode() 
#Episode:(states)
#c1 fb fb fb fb fb fb fb c1 fb c1 c2 c3 pub c2 c3 pass sleep
#Rewards:
#-2 -1 -1 -1 -1 -1 -1 -1 -2 -1 -2 -2 -2 1 -2 -2 10 0

#Episode:(states)
#c1 c2 c3 pass sleep
#Rewards:
#-2 -2 -2 10 0

#Episode:(states)
#c1 c2 c3 pub c3 pass sleep
#Rewards:
#-2 -2 -2 1 -2 10 0

# ==================== Question 8 ======================

def random_policy_episode():
           # 0,  1,    2,    3  , 4
    states=['c1','c2','c3','fb','sleep']
    #actions=['study','relax']
    #Reward for study:
    R1=[-2,-2,+10,0,0]
    #Reward for relax:
    R2=[-1,0,+1,-1,0]
    
    #transitional possibilities for study
    #      'c1','c2','c3','fb','sleep'
    P1=[[0  ,1  ,0    ,0   ,0],
        [0  ,0  ,1    ,0   ,0],
        [0  ,0  ,0    ,0   ,1],
        [1  ,0  ,0    ,0   ,0],
        [0  ,0  ,0    ,0   ,1]]        
    
    #transitional possibilities for relax
    #   'c1','c2','c3','fb','sleep'
    P2=[[0  ,0  ,0    ,1   ,0],
        [0  ,0  ,0    ,0   ,1],
        [0.2,0.4,0.4  ,0   ,0],
        [0  ,0  ,0    ,1   ,0],
        [0  ,0  ,0    ,0   ,1]]   
    
    myepisode='c1'
    s=0 #initial state is 0: c1
    r='-2'
    while s<4:
        x=random.uniform(0, 1)
        y=random.uniform(0, 1)
        cumulative_prob=0
        
        if x>0.5: #study
            for i in range(0,5):
                cumulative_prob+=P1[s][i]
                if y<cumulative_prob:
                    s=i   
                    myepisode=myepisode+' '+states[s]
                    r=r+' '+str(R1[s])
                    break
                
        else:#relax
            for i in range(0,5):
                cumulative_prob+=P2[s][i]
                if y<cumulative_prob:
                    s=i            
                    myepisode=myepisode+' '+states[s]
                    r=r+' '+str(R2[s])
                    break
                
    print("Episode:(states)")
    print(myepisode)
    print("Rewards:")
    print(r)  

random_policy_episode()
#Episode:(states)
#c1 fb fb c1 c2 sleep
#Rewards:
#-2 -1 -1 -2 -2 0
#
#Episode:(states)
#c1 c2 c3 sleep
#Rewards:
#-2 -2 10 0
#
#Episode:(states)
#c1 fb fb c1 fb fb fb c1 c2 c3 c3 c1 fb c1 fb c1 fb fb c1 fb fb c1 fb c1 fb fb c1 c2 c3 sleep
#Rewards:
#-2 -1 -1 -2 -1 -1 -1 -2 -2 10 1 -1 -1 -2 -1 -2 -1 -1 -2 -1 -1 -2 -1 -2 -1 -1 -2 -2 10 0


