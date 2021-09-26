#!/usr/bin/env python
# coding: utf-8

# In[1]:


import visdom
import time
import numpy as np


# In[2]:


class Visualizer(object):
    '''
    封装了visdom的基本操作，但是你仍然可以通过‘self.vis.function’
    调用原生的visdom接口
    '''
    def __init__(self,env="default",**kwargs):
        
        self.vis=visdom.Visdom(env=env,use_incoming_socket=False,**kwargs)
        
        #画的第几个数，相当于横坐标
        #保存("loss",23) 即loss的第23个点
        self.index={}
        self.log_text=""
        
    def reinit(self,env="default",**kwargs):
        '''
        修改visdom的配置
        '''
        self.vis=visdom.Visdom(env=env,use_incoming_socket=False,**kwargs)
        return self
    
    def plot_many(self,d):
        '''
        一次plot多个
        @params d: dict(name,value) i.e.("loss",0.11)
        '''
        for k,v in d.items():
            self.plot(k,v)
            
    
    def img_many(self,d):
        for k,v in d.items():
            self.img(k,v)
            
    
    def plot(self,name,y,**kwargs):
        '''
        self.plot("loss",1.00)
        '''
        x=self.index.get(name,0)
        self.vis.line(Y=np.array([y]),X=np.array([x]),
                     win=(name),
                     opts=dict(title=name),
                     update=None if x==0 else "append",
                     **kwargs
                     )
        
        self.index[name]=x+1
        
        
    def img(self,name,img_):
        '''
        self.img("input_img",t.Tensor(64,64))
        self.img("input_img",t.Tensor(3,64,64))
        self.img("input_img",t.Tensor(100,1,64,64))
        self.img("input_img",t.Tensor(100,3,64,64),nrows=10)
        '''
        if len(img_.size())<3:
            img_=img_.cpu().unsqueeze(0)
        self.vis.images(img_.cpu().numpy(),
                       win=(name),
                       opts=dict(title=name)
                      
                       )
        
    def img_grid_many(self,d):
        for k,v in d.items():
            self.img_grid(k,v)
            
            
    def img_grid(self,name,input_3d):
        '''
       一个batch的图片会变成一个网格图，i.e input(36,64,64)
       会变成6*6的网格图，每个格子大小为64*64
        
        '''
        self.img(name,tv.utils.make_grid(input_3d.cpu()[0].unsqueeze(1).clamp(max=1,min=0)))
        
        
    def log(self,info,win="log_text"):
        '''
        self.log({'loss':1,"lr":0.0001})
        '''
        self.log_text +=("[{time}]{info}<br>".format(time=time.strftime("%m%d_%H%M%S"),
                                                    info=info))
        
        self.vis.text(self.log_text,win)
        
        
    def __getattr__(self,name):
        
        return getattr(self.vis,name)
            
            


# In[ ]:




