#!/usr/bin/env python
# coding: utf-8

# In[ ]:


try:
    import ipdb
except:
    import pdb as ipdb
    
def sum(x):
    r=0
    for ii in x:
        r += ii
     
    return r

def mul(x):
    r=1
    
    for ii in x:
        r *=ii
    
    return r

ipdb.set_trace()

x=[1,2,3,4,5]
r=sum(x)
r=mul(x)


# In[ ]:




