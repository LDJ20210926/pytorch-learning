#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fire
def add(x,y):
    return x+y

def mul(**kwargs):
    a=kwargs["a"]
    b=kwargs["b"]
    return a*b
if __name__=="__main__":
    fire.Fire()


# In[ ]:




