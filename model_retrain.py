# -*- coding: utf-8 -*-
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from net.models import LeNet, LeNet_sparse
from net.quantization import apply_weight_sharing
import util

from scipy import sparse



import queue
import numpy as np
import collections
import pygraph as gone


def create_csr_graph(graph, num_vcount, ingestion_flag, dd):
    num_sources = 1
    num_thread = 4

    edge_dt = np.dtype([('src', np.int32), ('dst', np.int32), ('edgeid', np.int32)])
    csr_dt = np.dtype([('dst', np.int32), ('edgeid', np.int32)])
    offset_dt = np.dtype([('offset', np.int32)])

    outdir = ""
    # graph = gone.init(1, 1, outdir, num_sources, num_thread)  # Indicate one pgraph, and one vertex type
    tid0 = graph.init_vertex_type(num_vcount, True, "gtype") # initiate the vertex type
    pgraph = graph.create_schema(ingestion_flag, tid0, "friend", edge_dt) # initiate the pgraph

    # creating graph directly from file requires some efforts. Hope to fix that later
    manager = graph.get_pgraph_managerW(0) # This assumes single weighted graph, edgeid is the weight
    manager.add_edges(dd)  # ifile has no weights, edgeid will be generated
    pgraph.wait()  # You can't call add_edges() after wait(). The need of it will be removed in future.
    #snap_t = gone.create_static_view(pgraph, gone.enumView.eStale)
    
    offset_csr1, offset_csc1, nebrs_csr1, nebrs_csc1 = gone.create_csr_view(pgraph) 
    offset_csr = memoryview_to_np(offset_csr1, offset_dt) 
    offset_csc = memoryview_to_np(offset_csc1, offset_dt) 
    nebrs_csr = memoryview_to_np(nebrs_csr1, csr_dt) 
    nebrs_csc = memoryview_to_np(nebrs_csc1, csr_dt) 
    
    kernel_graph_flag = 0; #eADJ graph
    csr_graph = kernel.init_graph(offset_csr, nebrs_csr, offset_csc, nebrs_csc, kernel_graph_flag) 

    return csr_graph;



if __name__=="__main__":

	model = torch.load("/home/datalab/graphpy-workflow/model_compression/Deep-Compression-PyTorch/saves/initial_model.ptmodel")
	graph = gone.init(3, 3)
	managers = []
	datas = []
	names = []
	ingestion_flag = gone.enumGraph.eDdir;
	dim_list = []
	for name, param in model.named_parameters():
		if 'weight' in name:
			print("name: ", name)
			names.append(name)
			dict_tensor = {}
			tensor = param.data.cpu().numpy()
			dim_list.append(len(tensor[0]))
			num_vcount = max(len(tensor[0]), len(tensor))
			sparse_tensor = sparse.coo_matrix(tensor)
			index = np.row_stack((sparse_tensor.row, sparse_tensor.col))
			dd = np.zeros(len(sparse_tensor.row), dt); 
			edge_count = 0
			for i in range(len(sparse_tensor.row)): 
				dd[edge_count] = (sparse_tensor.row[i], sparse_tensor.col[i], edge_count)
				edge_count += 1

			manager = create_csr_graph(graph, num_vcount, ingestion_flag, dd)

			managers.append(manager)
			datas.append(sparse_tensor.data)
		else:
			dict[name] = param
	dict_weight = {}
	for i in range(len(managers)):
		gview = managers[i].create_static_view(gone.enumView.eStale)

		test_bfs(gview, 1)
		managers[i].run_bfs(1)
		dict_weight['data'] = data[i]
		dict_weight['gview'] = gview
	parser = argparse.ArgumentParser(description='PyTorch MNIST pruning from deep compression paper')
	parser.add_argument('--batch-size', type=int, default=50, metavar='N', help='input batch size for training (default: 50)')
	train_loader = torch.utils.data.DataLoader(
	datasets.MNIST('data', train=True, download=True,
	       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])), batch_size = args.batch_size, shuffle=True, **kwargs)

	test_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # feature = torch.tensor(feature)
    # train_id = torch.tensor(train_id)
    # test_id = torch.tensor(test_id)
    # train_y_label = torch.tensor(train_y_label)
    # test_y_label = torch.tensor(test_y_label)
    
	print("dim_list", dim_list)
	# train the network
	input_feature_dim = dim_list[0]
	hidden_layers = dim_list[1]
	num_class = dim_list[2]
	num_layers = len(managers)
	net = gcnconv.GCN(managers, input_feature_dim, hidden_layers, num_class, num_layers)
	optimizer = torch.optim.Adam(itertools.chain(net.parameters()), lr=0.01, weight_decay=5e-4)
	all_logits = []
	#print(input_X)
	#print("-------------------")
	start = datetime.datetime.now()
	for epoch in range(200):
	    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
	    for batch_idx, (data, target) in pbar:
	        data, target = data.to(device), target.to(device)
	        optimizer.zero_grad()
	        output = net.forward(data)
	        loss = F.nll_loss(output, target)
	        loss.backward()
	        optimizer.step()
	        if batch_idx % args.log_interval == 0:
	            done = batch_idx * len(data)
	            percentage = 100. * batch_idx / len(train_loader)
	            pbar.set_description(f'Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.6f}')




        #print('Epoch %d | Train_Loss: %.4f' % (epoch, loss.item()))

        # # check the accuracy for test data
        # logits_test = net.forward(feature)
        # logp_test = F.log_softmax(logits_test, 1)

        # acc_val = pubmed_util.accuracy(logp_test[test_id], test_y_label)
        #print('Epoch %d | Test_accuracy: %.4f' % (epoch, acc_val))

    # end = datetime.datetime.now()
    # difference = end - start
    # print("the time of graphpy is:", difference)
    # print('Epoch %d | Test_accuracy: %.4f' % (epoch, acc_val))




