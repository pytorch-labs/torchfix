import torch
a = torch.randn(5)
b = torch.randn(5)

# logsumexp
y = torch.log(torch.sum(torch.exp(x), 1, keepdim=True)) # has all the arguments for sum function call with keepdim as True
y = torch.log(torch.sum(torch.exp(2.5 + x), 1)) # addition operation inside the exp function call
y = torch.log(torch.sum(torch.exp(x),dim=1,keepdim=True)) # has all the arguments for sum function call
y = torch.log(torch.sum(torch.exp(x), dim=1)) #default value of keepdim is False
y = torch.log(torch.sum(torch.exp(x), dim=(1,2))) #default value of keepdim is False

# not logsumexp
y = torch.log(torch.sum(torch.exp(x), 1, keepdim=True) + 2.5) # cant have an addition operation inside the log function call 
y = torch.log(torch.sum(torch.exp(x) + 2.5, 1)) # Cant have an addition operation inside the sum function call with the argument as it expects a tensor
y = torch.log(2 + x) # missing sum and exp
y = torch.sum(torch.log(torch.exp(x)), 1) # not proper order of log and sum
y = torch.exp(torch.sum(torch.log(x), 1, keepdim=True)) #order of log,sum and exp is reversed
y = torch.log(torch.sum(torch.exp(2.5))) # this should not be flagged as the second argument is missing for sum function call and exp function call has an integer argument instead of a tensor
y = torch.log(torch.sum(torch.exp(x)), dim=1) #dim is not part of the sum fuction call
y = torch.log(torch.sum(torch.exp(x)), dim=None) #dim is not part of the sum fuction call and dim is None
y = torch.log(torch.sum(torch.exp(x), keepdim=True, dim=None)) #dim argument cannot be None 
y = torch.log(torch.sum(torch.exp(x), dim=(1,None))) #dim argument cannot be a tuple with None
y = torch.log(torch.sum(torch.exp(x), dim=(None,None))) #dim argument cannot be a tuple with None