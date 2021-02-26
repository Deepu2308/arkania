# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 04:46:10 2021

@author: deepu
"""

#import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

#output to action / updated model to ouput action directly
#get_action = lambda output: (output.cpu().detach().numpy() - .5)*2


class Coco(nn.Module):
    """
    class definition for BiPedal Walker
    
    Training stratergy
        Do mean centering of reward
        Get network output to be in the range (-1,1)
        Create Target of network as (network output * reward)
        Loss is now the MSE of (target - network output)        
    
    """
    def __init__(self, n_hidden = 800):
        super(Coco, self).__init__()


        self.observation_space = 169
        self.action_space      = 8
        self.n_hidden          = n_hidden
        self.DROPOUT           = .5        
        
        #fully connected layers
        self.fc1 = nn.Sequential(nn.Linear(self.observation_space, self.n_hidden),
                                 nn.Dropout(self.DROPOUT),
                                 nn.ReLU()
        )
        
        self.fc2 = nn.Sequential(nn.Linear(self.n_hidden,  self.action_space)
        )

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.fc2(x)
        
        return F.softmax(x, dim = 1)
    

class CocoConv(nn.Module):
    """
    class definition for BiPedal Walker
    
    Training stratergy
        Do mean centering of reward
        Get network output to be in the range (-1,1)
        Create Target of network as (network output * reward)
        Loss is now the MSE of (target - network output)        
    
    """
    def __init__(self, n_hidden = 800):
        super(CocoConv, self).__init__()


        self.observation_space = (13,13)
        self.action_space      = 8
        self.n_hidden          = n_hidden
        self.n_channels1       = 16
        
        #self.padding           = 2
        
        DROPOUT                = .5        
        
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,self.n_channels1, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(self.n_channels1),
            nn.ReLU())
        
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.n_channels1,self.n_channels1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.n_channels1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1))
                
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.n_channels1,self.n_channels1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.n_channels1),
            nn.ReLU(),
            nn.MaxPool2d(3,stride=1))
        
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.n_channels1,self.n_channels1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.n_channels1),
            nn.ReLU(),
            nn.MaxPool2d(3,stride=1))
        
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.n_channels1,self.n_channels1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.n_channels1),
            nn.ReLU(),
            nn.MaxPool2d(3,stride=1))
        
        
# =============================================================================
#         self.conv6 = nn.Sequential(
#             nn.Conv2d(self.n_channels1,self.n_channels1, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(self.n_channels1),
#             nn.ReLU(),
#             nn.MaxPool2d(3,stride=1),
#             nn.Dropout(DROPOUT))
#         
#         
#         self.conv7 = nn.Sequential(
#             nn.Conv2d(self.n_channels1,self.n_channels1, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(self.n_channels1),
#             nn.ReLU(),
#             nn.MaxPool2d(3,stride=1))
#         
#         
#         self.conv8 = nn.Sequential(
#             nn.Conv2d(self.n_channels1,self.n_channels1, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(self.n_channels1),
#             nn.ReLU(),
#             nn.MaxPool2d(3,stride=1))
#         
#         
#         self.conv9 = nn.Sequential(
#             nn.Conv2d(self.n_channels1,self.n_channels1, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(self.n_channels1),
#             nn.ReLU(),
#             nn.MaxPool2d(3,stride=1))
#         
#         
#         self.conv10 = nn.Sequential(
#             nn.Conv2d(self.n_channels1,self.n_channels1, kernel_size=3, stride=1, padding=0),
#             nn.BatchNorm2d(self.n_channels1),
#             nn.ReLU(),
#             nn.MaxPool2d(3,stride=1))
#         
#         
#         self.conv11 = nn.Sequential(
#             nn.Conv2d(self.n_channels1,self.n_channels1, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(self.n_channels1),
#             nn.ReLU(),
#             nn.Dropout(DROPOUT))
#         
# =============================================================================
        self.fc1 = nn.Sequential(nn.Linear(144, 4096),
            nn.ReLU(),
            nn.Dropout(DROPOUT))
        
        self.fc2 = nn.Sequential(nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(DROPOUT))
        
        self.fc3 = nn.Sequential(nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(DROPOUT))
        
        self.fc4 = nn.Sequential(nn.Linear(1024, self.action_space),
            nn.ReLU(),
            nn.Dropout(DROPOUT))
        
        
    def forward(self, x):
        
        #CONVOLUTIONAL LAYERS
        #print("Input",x.shape)
        out = self.conv1(x)
        #print("Conv1",out.shape)
        out = self.conv2(out)
        #print("Conv2",out.shape)
        out = self.conv3(out)
        #print("Conv3",out.shape)
        out = self.conv4(out)
        #print("Conv4",out.shape)
        out = self.conv5(out)
# =============================================================================
#         print("Conv5",out.shape)
#         out = self.conv6(out)
#         print("Conv6",out.shape)
#         out = self.conv7(out)
#         print("Conv7",out.shape)
#         out = self.conv8(out)
#         print("Conv8",out.shape)
#         out = self.conv9(out)
#         print("Conv9",out.shape)
#         out = self.conv10(out)
#         print("Conv10",out.shape)
#         out = self.conv11(out) 
#         print("Conv11",out.shape)
# =============================================================================

        #FLATTEN LAYER
        out = out.view(x.shape[0],-1)
        #print("Flatten",out.shape)
        
        #FULLY CONNECTED LAYERS
        out = self.fc1(out)
        #print("FC1",out.shape)
        out = self.fc2(out)
        #print("FC2",out.shape)
        out = self.fc3(out)
        #print("FC3",out.shape)
        out = self.fc4(out)
        #print("FC4",out.shape)
        
        #TENSOR OUT
        return F.softmax(out, dim = 1)