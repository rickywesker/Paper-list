V-AM-Softmax
import torch
import math
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
class SVAMLinear(nn.Module):                                                   
    def __init__(self,
                 in_channels,
                 num_class,
                 t = 1.2,                                                      
                 m = 0.35,
                 scale = 30):
        super(SVAMLinear,self).__init__()
        self.in_channels = in_channels                                         
        self.num_class = num_class
        self.t = t 
        self.m = m
        self.scale = scale 
        self.weight = Parameter(torch.Tensor(num_class, in_channels))          
        self.reset_parameters()                                                
        self.register_parameter('bias',None)                                   
        #self.weights.data.uniform_(-1,1).renorm(2,1,1e-5).mul(1e5)            
            
            
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))                             
        self.weight.data.uniform_(-stdv,stdv)                                  
                
    def forward(self,input,target):                                            
        #norm_weights = F.normalize(self.weights,dim=0)                        
        ex = input / torch.norm(input, 2, 1, keepdim=True)
        ew = self.weight / torch.norm(self.weight,2,1,keepdim=True)            
        cos_theta = torch.mm(ex, ew.t())                                       
        batch_size = target.size(0)
        gtScore = cos_theta[torch.arange(0,batch_size),target].view(-1,1)      
        mask = cos_theta > (gtScore - self.m)                                  
        finalScore = torch.where(gtScore > self.m, gtScore - self.m,gtScore)   
        hardEx = cos_theta[mask]                                               
        cos_theta[mask] = self.t * hardEx + self.t - 1.0                       
        cos_theta.scatter_(1, target.data.view(-1,1),finalScore)
        cos_theta *= self.scale                                                
        return cos_theta                                                       
