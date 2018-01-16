"""
Note: adapted from the original debugging environment to have Box obs space

Simple environment with known optimal policy and value function.

This environment has just two actions.
Action 0 yields 0 reward and then terminates the session.
Action 1 yields 1 reward and then terminates the session.

Optimal policy: action 1.

Optimal value function: v(0)=1 (there is only one state, state 0)
"""

import numpy as np
import gym
from gym import spaces
import pandas as pd
from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.finance as mpl_finance
import skimage
from skimage import transform
from gym.utils import seeding

class OneRoundDeterministicRewardBoxObsEnv(gym.Env):
   
    metadata = {
        'render.modes':[],
        'semantics.autoreset':""
    }
    
    def __init__(self, obs_shape=(64,64,1)):
        self.dir = 'D:\PythonProject\\untiled2\data\stock.csv'  
        self.period = 50
        self.idx = self.period
        self.actions = [1, 0]  # self.actions = ["LONG","SHORT"]
        self.start = 0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(0,500, shape = obs_shape)
        self.metadata = {'render.modes': ['human']}
        self.stack = 1
        self.data = pd.read_csv(self.dir, header=None, sep="\t") 

    def _step(self, action):
        #trading start at the n+1 bar, each observation is looking back n bars' OHLC data (50*4)'s 
    #rgb pixel data. The only difference with TradingEnv_v0 is the observation. 
        opens = self.data.iloc[self.start : self.idx ,1].as_matrix()
        highs = self.data.iloc[self.start : self.idx ,2].as_matrix()
        lows = self.data.iloc[self.start : self.idx ,3].as_matrix()
        closes = self.data.iloc[self.start : self.idx ,4].as_matrix()
    
        # We would draw the chart first and then get the rgb data 
        fig = figure.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)
    
        mpl_finance.candlestick2_ohlc(ax, opens, highs, lows, closes, width=0.6, colorup='w', colordown='k', alpha=1)
        ax.set_axis_off()
        
        fig.canvas.draw()   
        ncols, nrows = fig.canvas.get_width_height()
        observation = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(nrows, ncols, 3)
        observation = skimage.transform.resize(observation,(64,64))  
        observation = np.delete(observation,np.s_[1:4], axis = 2)  
        observation = np.reshape(observation,(64,64,1))   # a is the final shrinked image data we need 
    
        done = True
    #for this env, all we need is reward, no need to set up portfolio
        info = {}
        reward = 0    
    
    #reward is calculated with numerical data, aside from the pixel observation
        if action == 1:    #1 is long
          gain= self.data.iloc[self.idx:(self.idx+50),1:5].as_matrix().max() - self.data.iloc[self.idx,4]
          loss= self.data.iloc[self.idx:(self.idx+50),1:5].as_matrix().min() - self.data.iloc[self.idx,4]
          reward = (abs(gain) - abs(loss))*10000
       
        elif action == 0:  #0 is short
          gain= self.data.iloc[self.idx,4] - self.data.iloc[self.idx:(self.idx+50),1:5].as_matrix().min()  
          loss= self.data.iloc[self.idx,4] - self.data.iloc[self.idx:(self.idx+50),1:5].as_matrix().max() 
          reward = (abs(gain) - abs(loss))*10000
         
        self.idx += self.stack    
        self.start += self.stack        
    
        info["trade No."+str(self.idx)] = self.actions[action]
        
        return observation, reward, done , info  

    def _reset(self):
        opens = self.data.iloc[self.start : self.idx ,1].as_matrix()
        highs = self.data.iloc[self.start : self.idx ,2].as_matrix()
        lows = self.data.iloc[self.start : self.idx ,3].as_matrix()
        closes = self.data.iloc[self.start : self.idx ,4].as_matrix()
    
        # We would draw the chart first and then get the rgb data 
        fig = figure.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)
    
        mpl_finance.candlestick2_ohlc(ax, opens, highs, lows, closes, width=0.6, colorup='w', colordown='k', alpha=1)
        ax.set_axis_off()
        
        fig.canvas.draw()   
        ncols, nrows = fig.canvas.get_width_height()
        observation = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(nrows, ncols, 3)
        observation = skimage.transform.resize(observation,(64,64))  
        observation = np.delete(observation,np.s_[1:4], axis = 2)  
        observation = np.reshape(observation,(64,64,1))   # a is the final shrinked image data we need 
        
        return observation
    
    def _render(self, mode='human', close=False):
        pass
        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
