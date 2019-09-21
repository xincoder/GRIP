import argparse
import os 
import sys
import numpy as np 
import torch
import torch.optim as optim
from model import Model
from xin_feeder_baidu import Feeder
from datetime import datetime
import random
import itertools

CUDA_VISIBLE_DEVICES='1'
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

def seed_torch(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch()

max_x = 1. 
max_y = 1. 
history_frames = 6 # 3 second * 2 frame/second
future_frames = 6 # 3 second * 2 frame/second

batch_size_train = 64 
batch_size_val = 32
batch_size_test = 1
total_epoch = 50
base_lr = 0.01
lr_decay_epoch = 5
dev = 'cuda:0' 
work_dir = './trained_models'
log_file = os.path.join(work_dir,'log_test.txt')
test_result_file = 'prediction_result.txt'

criterion = torch.nn.SmoothL1Loss()

if not os.path.exists(work_dir):
	os.makedirs(work_dir)

def my_print(pra_content):
	with open(log_file, 'a') as writer:
		print(pra_content)
		writer.write(pra_content+'\n')

def display_result(pra_results, pra_pref='Train_epoch'):
	all_overall_sum_list, all_overall_num_list = pra_results
	overall_sum_time = np.sum(all_overall_sum_list**0.5, axis=0)
	overall_num_time = np.sum(all_overall_num_list, axis=0)
	overall_loss_time = (overall_sum_time / overall_num_time) 
	overall_log = '|{}|[{}] All_All: {}'.format(datetime.now(), pra_pref, ' '.join(['{:.3f}'.format(x) for x in list(overall_loss_time) + [np.sum(overall_loss_time)]]))
	my_print(overall_log)
	return overall_loss_time
	

def my_save_model(pra_model, pra_epoch):
	path = '{}/model_epoch_{:04}.pt'.format(work_dir, pra_epoch)
	torch.save(
		{
			'xin_graph_seq2seq_model': pra_model.state_dict(),
		}, 
		path)
	print('Successfull saved to {}'.format(path))


def my_load_model(pra_model, pra_path):
	checkpoint = torch.load(pra_path)
	pra_model.load_state_dict(checkpoint['xin_graph_seq2seq_model'])
	print('Successfull loaded from {}'.format(pra_path))
	return pra_model


def data_loader(pra_path, pra_batch_size=128, pra_shuffle=False, pra_drop_last=False, train_val_test='train'):
	feeder = Feeder(data_path=pra_path, graph_args=graph_args, train_val_test=train_val_test)
	loader = torch.utils.data.DataLoader(
		dataset=feeder,
		batch_size=pra_batch_size,
		shuffle=pra_shuffle,
		drop_last=pra_drop_last, 
		num_workers=10,
		)
	return loader
	
def preprocess_data(pra_data, pra_rescale_xy):
	# pra_data: (N, C, T, V)
	# C = 11: [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading] + [mask]	
	feature_id = [3, 4, 9, 10]
	ori_data = pra_data[:,feature_id].detach()
	data = ori_data.detach().clone()

	new_mask = (data[:, :2, 1:]!=0) * (data[:, :2, :-1]!=0) 
	data[:, :2, 1:] = (data[:, :2, 1:] - data[:, :2, :-1]).float() * new_mask.float()
	data[:, :2, 0] = 0	

	# # small vehicle: 1, big vehicles: 2, pedestrian 3, bicycle: 4, others: 5
	object_type = pra_data[:,2:3]

	data = data.float().to(dev)
	ori_data = ori_data.float().to(dev)
	object_type = object_type.to(dev) #type
	data[:,:2] = data[:,:2] / pra_rescale_xy

	return data, ori_data, object_type
	

def compute_RMSE(pra_pred, pra_GT, pra_mask, pra_error_order=2):
	pred = pra_pred * pra_mask # (N, C, T, V)=(N, 2, 6, 120)
	GT = pra_GT * pra_mask # (N, C, T, V)=(N, 2, 6, 120)
	
	x2y2 = torch.sum(torch.abs(pred - GT)**pra_error_order, dim=1) # x^2+y^2, (N, C, T, V)->(N, T, V)=(N, 6, 120)
	overall_sum_time = x2y2.sum(dim=-1) # (N, T, V) -> (N, T)=(N, 6)
	overall_mask = pra_mask.sum(dim=1).sum(dim=-1) # (N, C, T, V) -> (N, T)=(N, 6)
	overall_num = overall_mask 

	return overall_sum_time, overall_num, x2y2


def train_model(pra_model, pra_data_loader, pra_optimizer, pra_epoch_log):
	# pra_model.to(dev)
	pra_model.train()
	rescale_xy = torch.ones((1,2,1,1)).to(dev)
	rescale_xy[:,0] = max_x
	rescale_xy[:,1] = max_y

	# train model using training data
	for iteration, (ori_data, A, _) in enumerate(pra_data_loader):
		# print(iteration, ori_data.shape, A.shape)
		# ori_data: (N, C, T, V)
		# C = 11: [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading] + [mask]
		data, no_norm_loc_data, object_type = preprocess_data(ori_data, rescale_xy)
		for now_history_frames in range(1, data.shape[-2]):
			input_data = data[:,:,:now_history_frames,:] # (N, C, T, V)=(N, 4, 6, 120)
			output_loc_GT = data[:,:2,now_history_frames:,:] # (N, C, T, V)=(N, 2, 6, 120)
			output_mask = data[:,-1:,now_history_frames:,:] # (N, C, T, V)=(N, 1, 6, 120)
			
			A = A.float().to(dev)
		
			predicted = pra_model(pra_x=input_data, pra_A=A, pra_pred_length=output_loc_GT.shape[-2], pra_teacher_forcing_ratio=0, pra_teacher_location=output_loc_GT) # (N, C, T, V)=(N, 2, 6, 120)
			
			########################################################
			# Compute loss for training
			########################################################
			# We use abs to compute loss to backward update weights
			# (N, T), (N, T)
			overall_sum_time, overall_num, _ = compute_RMSE(predicted, output_loc_GT, output_mask, pra_error_order=1)
			# overall_loss
			total_loss = torch.sum(overall_sum_time) / torch.max(torch.sum(overall_num), torch.ones(1,).to(dev)) #(1,)

			now_lr = [param_group['lr'] for param_group in pra_optimizer.param_groups][0]
			my_print('|{}|{:>20}|\tIteration:{:>5}|\tLoss:{:.8f}|lr: {}|'.format(datetime.now(), pra_epoch_log, iteration, total_loss.data.item(),now_lr))
			
			pra_optimizer.zero_grad()
			total_loss.backward()
			pra_optimizer.step()

		

def val_model(pra_model, pra_data_loader):
	# pra_model.to(dev)
	pra_model.eval()
	rescale_xy = torch.ones((1,2,1,1)).to(dev)
	rescale_xy[:,0] = max_x
	rescale_xy[:,1] = max_y
	all_overall_sum_list = []
	all_overall_num_list = []

	all_car_sum_list = []
	all_car_num_list = []
	all_human_sum_list = []
	all_human_num_list = []
	all_bike_sum_list = []
	all_bike_num_list = []
	# train model using training data
	for iteration, (ori_data, A, _) in enumerate(pra_data_loader):
		# data: (N, C, T, V)
		# C = 11: [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading] + [mask]
		data, no_norm_loc_data, _ = preprocess_data(ori_data, rescale_xy)

		for now_history_frames in range(6, 7):
			input_data = data[:,:,:now_history_frames,:] # (N, C, T, V)=(N, 4, 6, 120)
			output_loc_GT = data[:,:2,now_history_frames:,:] # (N, C, T, V)=(N, 2, 6, 120)
			output_mask = data[:,-1:,now_history_frames:,:] # (N, C, T, V)=(N, 1, 6, 120)

			ori_output_loc_GT = no_norm_loc_data[:,:2,now_history_frames:,:]
			ori_output_last_loc = no_norm_loc_data[:,:2,now_history_frames-1:now_history_frames,:]

			# for category
			cat_mask = ori_data[:,2:3, now_history_frames:, :] # (N, C, T, V)=(N, 1, 6, 120)
			
			A = A.float().to(dev)
			predicted = pra_model(pra_x=input_data, pra_A=A, pra_pred_length=output_loc_GT.shape[-2], pra_teacher_forcing_ratio=0, pra_teacher_location=output_loc_GT) # (N, C, T, V)=(N, 2, 6, 120)
			########################################################
			# Compute details for training
			########################################################
			predicted = predicted*rescale_xy
			# output_loc_GT = output_loc_GT*rescale_xy

			for ind in range(1, predicted.shape[-2]):
				predicted[:,:,ind] = torch.sum(predicted[:,:,ind-1:ind+1], dim=-2)
			predicted += ori_output_last_loc

			### overall dist
			# overall_sum_time, overall_num, x2y2 = compute_RMSE(predicted, output_loc_GT, output_mask)		
			overall_sum_time, overall_num, x2y2 = compute_RMSE(predicted, ori_output_loc_GT, output_mask)		
			# all_overall_sum_list.extend(overall_sum_time.detach().cpu().numpy())
			all_overall_num_list.extend(overall_num.detach().cpu().numpy())
			# x2y2 (N, 6, 39)
			now_x2y2 = x2y2.detach().cpu().numpy()
			now_x2y2 = now_x2y2.sum(axis=-1)
			all_overall_sum_list.extend(now_x2y2)

			### car dist
			car_mask = (((cat_mask==1)+(cat_mask==2))>0).float().to(dev)
			car_mask = output_mask * car_mask
			car_sum_time, car_num, car_x2y2 = compute_RMSE(predicted, ori_output_loc_GT, car_mask)		
			all_car_num_list.extend(car_num.detach().cpu().numpy())
			# x2y2 (N, 6, 39)
			car_x2y2 = car_x2y2.detach().cpu().numpy()
			car_x2y2 = car_x2y2.sum(axis=-1)
			all_car_sum_list.extend(car_x2y2)

			### human dist
			human_mask = (cat_mask==3).float().to(dev)
			human_mask = output_mask * human_mask
			human_sum_time, human_num, human_x2y2 = compute_RMSE(predicted, ori_output_loc_GT, human_mask)		
			all_human_num_list.extend(human_num.detach().cpu().numpy())
			# x2y2 (N, 6, 39)
			human_x2y2 = human_x2y2.detach().cpu().numpy()
			human_x2y2 = human_x2y2.sum(axis=-1)
			all_human_sum_list.extend(human_x2y2)

			### bike dist
			bike_mask = (cat_mask==4).float().to(dev)
			bike_mask = output_mask * bike_mask
			bike_sum_time, bike_num, bike_x2y2 = compute_RMSE(predicted, ori_output_loc_GT, bike_mask)		
			all_bike_num_list.extend(bike_num.detach().cpu().numpy())
			# x2y2 (N, 6, 39)
			bike_x2y2 = bike_x2y2.detach().cpu().numpy()
			bike_x2y2 = bike_x2y2.sum(axis=-1)
			all_bike_sum_list.extend(bike_x2y2)

	
	result_car = display_result([np.array(all_car_sum_list), np.array(all_car_num_list)], pra_pref='car')
	result_human = display_result([np.array(all_human_sum_list), np.array(all_human_num_list)], pra_pref='human')
	result_bike = display_result([np.array(all_bike_sum_list), np.array(all_bike_num_list)], pra_pref='bike')

	result = 0.20*result_car + 0.58*result_human + 0.22*result_bike
	overall_log = '|{}|[{}] All_All: {}'.format(datetime.now(), 'WS', ' '.join(['{:.3f}'.format(x) for x in list(result) + [np.sum(result)]]))
	my_print(overall_log)

	all_overall_sum_list = np.array(all_overall_sum_list)
	all_overall_num_list = np.array(all_overall_num_list)
	return all_overall_sum_list, all_overall_num_list



def test_model(pra_model, pra_data_loader):
	# pra_model.to(dev)
	pra_model.eval()
	rescale_xy = torch.ones((1,2,1,1)).to(dev)
	rescale_xy[:,0] = max_x
	rescale_xy[:,1] = max_y
	all_overall_sum_list = []
	all_overall_num_list = []
	with open(test_result_file, 'w') as writer:
		# train model using training data
		for iteration, (ori_data, A, mean_xy) in enumerate(pra_data_loader):
			# data: (N, C, T, V)
			# C = 11: [frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading] + [mask]
			data, no_norm_loc_data, _ = preprocess_data(ori_data, rescale_xy)
			input_data = data[:,:,:history_frames,:] # (N, C, T, V)=(N, 4, 6, 120)
			output_mask = data[:,-1,-1,:] # (N, V)=(N, 120)
			# print(data.shape, A.shape, mean_xy.shape, input_data.shape)

			ori_output_last_loc = no_norm_loc_data[:,:2,history_frames-1:history_frames,:]
		
			A = A.float().to(dev)
			predicted = pra_model(pra_x=input_data, pra_A=A, pra_pred_length=future_frames, pra_teacher_forcing_ratio=0, pra_teacher_location=None) # (N, C, T, V)=(N, 2, 6, 120)
			predicted = predicted *rescale_xy 

			for ind in range(1, predicted.shape[-2]):
				predicted[:,:,ind] = torch.sum(predicted[:,:,ind-1:ind+1], dim=-2)
			predicted += ori_output_last_loc

			now_pred = predicted.detach().cpu().numpy() # (N, C, T, V)=(N, 2, 6, 120)
			now_mean_xy = mean_xy.detach().cpu().numpy() # (N, 2)
			now_ori_data = ori_data.detach().cpu().numpy() # (N, C, T, V)=(N, 11, 6, 120)
			now_mask = now_ori_data[:, -1, -1, :] # (N, V)
			
			now_pred = np.transpose(now_pred, (0, 2, 3, 1)) # (N, T, V, 2)
			now_ori_data = np.transpose(now_ori_data, (0, 2, 3, 1)) # (N, T, V, 11)
			
			# print(now_pred.shape, now_mean_xy.shape, now_ori_data.shape, now_mask.shape)

			for n_pred, n_mean_xy, n_data, n_mask in zip(now_pred, now_mean_xy, now_ori_data, now_mask):
				# (6, 120, 2), (2,), (6, 120, 11), (120, )
				num_object = np.sum(n_mask).astype(int)
				# only use the last time of original data for ids (frame_id, object_id, object_type)
				# (6, 120, 11) -> (num_object, 3)
				n_dat = n_data[-1, :num_object, :3].astype(int)
				for time_ind, n_pre in enumerate(n_pred[:, :num_object], start=1):
					# (120, 2) -> (n, 2)
					# print(n_dat.shape, n_pre.shape)
					for info, pred in zip(n_dat, n_pre+n_mean_xy):
						information = info.copy()
						information[0] = information[0] + time_ind
						result = ' '.join(information.astype(str)) + ' ' + ' '.join(pred.astype(str)) + '\n'
						# print(result)
						writer.write(result)


def run_trainval(pra_model, pra_traindata_path, pra_testdata_path):
	loader_train = data_loader(pra_traindata_path, pra_batch_size=batch_size_train, pra_shuffle=True, pra_drop_last=True, train_val_test='train')
	loader_test = data_loader(pra_testdata_path, pra_batch_size=batch_size_train, pra_shuffle=True, pra_drop_last=True, train_val_test='all')

	# evaluate on testing data (observe 5 frame and predict 1 frame)
	loader_val = data_loader(pra_traindata_path, pra_batch_size=batch_size_val, pra_shuffle=False, pra_drop_last=False, train_val_test='val') 
	
	optimizer = optim.Adam(
		[{'params':model.parameters()},],) # lr = 0.0001)
		
	for now_epoch in range(total_epoch):
		all_loader_train = itertools.chain(loader_train, loader_test)
		
		my_print('#######################################Train')
		train_model(pra_model, all_loader_train, pra_optimizer=optimizer, pra_epoch_log='Epoch:{:>4}/{:>4}'.format(now_epoch, total_epoch))
		
		my_save_model(pra_model, now_epoch)

		my_print('#######################################Test')
		display_result(
			val_model(pra_model, loader_val),
			pra_pref='{}_Epoch{}'.format('Test', now_epoch)
		)


def run_test(pra_model, pra_data_path):
	loader_test = data_loader(pra_data_path, pra_batch_size=batch_size_test, pra_shuffle=False, pra_drop_last=False, train_val_test='test')
	test_model(pra_model, loader_test)



if __name__ == '__main__':
	graph_args={'max_hop':2, 'num_node':120}
	model = Model(in_channels=4, graph_args=graph_args, edge_importance_weighting=True)
	model.to(dev)

	# train and evaluate model
	run_trainval(model, pra_traindata_path='./train_data.pkl', pra_testdata_path='./test_data.pkl')
	
	# pretrained_model_path = './trained_models/model_epoch_0016.pt'
	# model = my_load_model(model, pretrained_model_path)
	# run_test(model, './test_data.pkl')
	
		
		

