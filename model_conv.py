import torch as th
import torch.nn as nn
import torch.utils.dlpack
import pygraph as gone
import torch.nn.functional as F
import gp_apis




class Spmmw_model(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, X, weight,  num_vcount, dim, bias):
        dim_X = X.size(1)
        dim_weight = weight.size(1)
        num_ecount = weight.size(0)
        #non_bias = torch.zeros(bias.shape)

        # add the bias tenosr
        #bias_tensor = []
        #for i in range(X.size(0)):
        #    bias_tensor.append(bias)
        #bias_tensor1 = th.Tensor(bias_tensor)
        #bias_tensor1 = bias_tensor1.reshape(num_vcount, dim)

        res = gp_apis.gp_spmmw_model(graph, X, weight, bias, num_vcount, dim, gone.enumOP.eSUM, 0)  # not sure

        ctx.backward_cache = graph, num_vcount, dim, weight, X, dim_X, dim_weight, num_ecount, bias
        #res = torch.add(res, bias)

        return res

    @staticmethod
    def backward(ctx, dz):
        graph, num_vcount, dim, weight, x, dim_x, dim_weight, num_ecount, bias = ctx.backward_cache
     
        x_res = gp_apis.gp_spmmw_model_without_bias(graph, dz, weight, num_vcount, dim_x, gone.enumOP.eSUM, 1)  # not sure
        weight_res = gp_apis.gp_sddmm_model(graph, x, dz, num_ecount, dim_weight, gone.enumOP.eMUL)
        bias_res = torch.mean(dz, 0)
        #print("bias_back",bias_res)
        #print("X is NAN?", torch.isnan(x_res).any())
        #print("fea is NAN?", torch.isnan(feat).any())

        return None, x_res, weight_res, None, None, bias_res



def run_gspmm_model(graph, X, weight, num_vcount, dim, bias):
    #bias_tensor = th.Tensor(bias)
    return Spmmw_model.apply(graph, X, weight, num_vcount, dim, bias)





"""

class ModelcompressConv(nn.Module):
    def __init__(self,
                 graph,
                 in_feats,
                 out_feats, weight, bias):
        super(ModelcompressConv, self).__init__() 
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.graph = graph
        self.weight = weight
        self.bias = bias
        


    def forward(self, feat):
        num_vcount = feat.size(0)
        dim = self._out_feats
        rst = run_gspmm_model(self.graph, feat, self.weight, self.bias, num_vcount, dim, self.bias)
        #print(rst)
        return rst

"""






class ModelcompressConv(nn.Module):
    def __init__(self,
                 graph,
                 in_feats, weight,
                 out_feats, bias):
        super(ModelcompressConv, self).__init__() 
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.graph = graph
        #self.weight = weight
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        


    def forward(self, feat):
        num_vcount = feat.size(0)
        dim = self._out_feats
        #print("weight is NAN?", torch.isnan(self.weight).any())
        #print("fea is NAN?", torch.isnan(feat).any())
        rst = run_gspmm_model(self.graph, feat, self.weight, num_vcount, dim, self.bias)
        #print(rst)
        return rst



class Compressed_Model(nn.Module):
    def __init__(self,
                 graph,
                 in_feats,
                 n_hidden1,
                 n_hidden2,
                 n_classes,
                 n_layers, weight_list, bias_list):
        super(Compressed_Model, self).__init__()
        self.graph = graph
        self.layers = nn.ModuleList()
        self.weight_list = weight_list
        self.bias_list = bias_list
        #print("ha")
        #print(weight_list[0].size())
        #print("weight size" + "0 is:", in_feats, n_hidden1)

        # input layer
        self.layers.append(ModelcompressConv(graph = self.graph[0],in_feats = in_feats, out_feats = n_hidden1, weight = self.weight_list[0], bias = self.bias_list[0]))
        #self.layers.append(ModelcompressConv(graph = self.graph[0],in_feats = in_feats, out_feats = n_hidden, bias = self.bias_list[0]))

        # hidden layers
        for i in range(1, n_layers - 1):
            #print("ha")
            #print(self.weight_list[i].size())
            #print("weight size" + str(i) + " is:", n_hidden1, n_hidden2)
            self.layers.append(ModelcompressConv(graph = self.graph[i], in_feats = n_hidden1, out_feats = n_hidden2, weight = self.weight_list[i], bias = self.bias_list[i]))
            #self.layers.append(ModelcompressConv(graph = self.graph[i], in_feats = n_hidden, out_feats = n_hidden, bias = self.bias_list[i]))

        # output layer
        #print("ha")
        #print(weight_list[n_layers - 1].size())
        #print("weight size" + str(2) + " is:", n_hidden2, n_classes)

        self.layers.append(ModelcompressConv(graph = self.graph[n_layers - 1], in_feats = n_hidden2, out_feats = n_classes, weight = self.weight_list[n_layers - 1], bias = self.bias_list[n_layers - 1]))
        #self.layers.append(ModelcompressConv(graph = self.graph[n_layers - 1], in_feats = n_hidden, out_feats = n_classes, bias = self.bias_list[n_layers - 1]))


    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            #print("layers" + str(i))
            #if i != 0:
                #h = self.dropout(h)
            h = layer(h)
            h = F.relu(h)
            #print("result")
            #print (h)
            #print("is result contains nan?", torch.isnan(h).any())
        h = F.log_softmax(h, dim=1)
        return h






"""
    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            print("layers" + str(i))
            #if i != 0:
                #h = self.dropout(h)
            h = layer(h)
            print("result")
            print (h.size())
        h = F.log_softmax(h, dim=1)
        return h
"""






