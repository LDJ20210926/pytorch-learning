#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import torch as t


# In[4]:


class DefaultConfig(object):
    env="default"  #visdom 环境
    vis_port=8097 #visdom 端口
    model="SqueezeNet" #使用的模型，名字必须与models/__init__.py中的名字一致
    
    train_data_root="./data/train/" #训练集存放路径
    test_data_root="./data/test1/" #测试集存放路径
    load_model_path=None #加载预训练的模型的路径，为None代表不加载
    
    batch_size=32 
    use_gpu=True 
    num_workers=2
    print_freq=20
    
    debug_file="./tmp/"
    result_file="result.csv"
    
    max_epoch=10
    lr=0.001
    lr_decay=0.5
    weight_decay=1e-5 #损失函数
    
    def _parse(self,kwargs):
        '''
        根据字典kwargs更新config参数
        '''
        for k,v in kwargs.items():
            if not hasattr(self,k):
                
                warnings.warn("Warning: opt has not attribut %s" % k )
            setattr(self,k,v)
            
        opt.device=t.device("cuda") if opt.use_gpu else t.device("cpu")
        
        print("user config:")
        for k,v in self.__class__.__dict__.items():
            if not k.startswith("_"):
                print(k,getattr(self,k))
                
                
opt=DefaultConfig()
       


# In[5]:


#try:   
    #get_ipython().system('jupyter nbconvert --to python config.ipynb')
    # python即转化为.py，script即转化为.html
    # file_name.ipynb即当前module的文件名
#except:
    #pass


# In[ ]:




