from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import time
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dst
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from get_dataset import get_data,NpyFlowDataset
from utils import AverageMeter, accuracy, transform_time
from utils import load_pretrained_model, save_checkpoint
from utils import create_exp_dir, count_parameters_in_MB
from net import TeacherMLP
from network import define_tsnet

parser = argparse.ArgumentParser(description='train base net')

# various path
parser.add_argument('--save_root', type=str, default='./results/base2017_few', help='models and logs are saved here')
parser.add_argument('--img_root', type=str, default='./CIC17_18/CIC17_18', help='path name of image dataset')

# training hyper parameters
parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
parser.add_argument('--epochs', type=int, default=200, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=512, help='The size of batch')
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
#parser.add_argument('--num_class', type=int, default=11, help='number of classes')
parser.add_argument('--cuda', type=int, default=1)

# others
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--note', type=str, default='try', help='note for this run')

# net and dataset choosen
parser.add_argument('--dataset', type=str, default='cicids-2018', help='Dataset name, defining in get_data.py')
parser.add_argument('--selected_class', type=list, default=[0,1],
					help='List of known attack classes for training, defines the subset of attack classes to be used.')
parser.add_argument('--attack_max_samples', type=int, default=500,
					help='Maximum number of samples per attack class in the dataset.')
parser.add_argument('--test_split_size', type=float, default=0.8, help='Split size for the test dataset, used in few-shot learning settings, when few-shot setting, should large.')
#parser.add_argument('--data_name', type=str, default='cicids-2018', help='name of dataset') # cifar10/cifar100
parser.add_argument('--net_name', type=str, default='TeacherMLP', help='name of basenet')  # resnet20/resnet110


#args, unparsed = parser.parse_known_args()
args = parser.parse_args()

args.save_root = os.path.join(args.save_root, args.note)
create_exp_dir(args.save_root)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.save_root, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list, num_classes, feature_num, attack_classes=get_data(args)
	print(num_classes)
	print(feature_num)
	# if args.cuda:
	# 	torch.cuda.manual_seed(args.seed)
	# 	cudnn.enabled = True
	# 	cudnn.benchmark = True
	logging.info("args = %s", args)
	#logging.info("unparsed_args = %s", unparsed)

	logging.info('----------- Network Initialization --------------')
	net = TeacherMLP(feature_num,num_classes)#修改
	logging.info('%s', net)
	logging.info("param size = %fMB", count_parameters_in_MB(net))
	logging.info('-----------------------------------------------')

	# save initial parameters
	logging.info('Saving initial parameters......') 
	save_path = os.path.join(args.save_root, 'initial_r{}.pth.tar'.format(args.net_name[6:]))
	torch.save({
		'epoch': 0,
		'net': net.state_dict(),
		'prec@1': 0.0,
	}, save_path)

	# initialize optimizer
	optimizer = torch.optim.SGD(net.parameters(),
								lr = args.lr, 
								momentum = args.momentum, 
								weight_decay = args.weight_decay,
								nesterov = True)

	# define loss functions
	# if args.cuda:
	# 	criterion = torch.nn.CrossEntropyLoss().cuda()
	# else:
	criterion = torch.nn.CrossEntropyLoss()

	# define transforms
	# if args.data_name == 'cifar10':
	# 	dataset = dst.CIFAR10
	# 	mean = (0.4914, 0.4822, 0.4465)
	# 	std  = (0.2470, 0.2435, 0.2616)
	# elif args.data_name == 'cifar100':
	# 	dataset = dst.CIFAR100
	# 	mean = (0.5071, 0.4865, 0.4409)
	# 	std  = (0.2673, 0.2564, 0.2762)
	# else:
	# 	raise Exception('Invalid dataset name...')
	#
	# train_transform = transforms.Compose([
	# 		transforms.Pad(4, padding_mode='reflect'),
	# 		transforms.RandomCrop(32),
	# 		transforms.RandomHorizontalFlip(),
	# 		transforms.ToTensor(),
	# 		transforms.Normalize(mean=mean,std=std)
	# 	])
	# test_transform = transforms.Compose([
	# 		transforms.CenterCrop(32),
	# 		transforms.ToTensor(),
	# 		transforms.Normalize(mean=mean,std=std)
	# 	])
	#
	# # define data loader
	# train_loader = torch.utils.data.DataLoader(#修改
	# 		dataset(root      = args.img_root,
	# 				transform = train_transform,
	# 				train     = True,
	# 				download  = True),
	# 		batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
	# test_loader = torch.utils.data.DataLoader(
	# 		dataset(root      = args.img_root,
	# 				transform = test_transform,
	# 				train     = False,
	# 				download  = True),
	# 		batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

	train_loader=torch.utils.data.DataLoader(
		NpyFlowDataset(X_train,y_train),
		batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
	)

	test_loader=torch.utils.data.DataLoader(
		NpyFlowDataset(X_test, y_test),
		batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
	)

	best_top1 = 0
	best_f1=0
	best_precision=0
	best_recall=0
	best_tp=0
	best_fp=0
	best_tn=0
	best_fn=0
	best_accuracy=0
	top1 = 0
	f1=0
	precision=0
	recall=0
	tp=0
	fp=0
	tn=0
	fn=0
	accuracy=0
	for epoch in range(1, args.epochs+1):
		adjust_lr(optimizer, epoch)

		# train one epoch
		epoch_start_time = time.time()
		train(train_loader, net, optimizer, criterion, epoch)

		# evaluate on testing set
		logging.info('Testing the models......')
		test_top1,f1,precision,recall,accuracy1,tp,fp,tn,fn = test(test_loader, net, criterion)

		epoch_duration = time.time() - epoch_start_time
		logging.info('Epoch time: {}s'.format(int(epoch_duration)))

		# save model
		is_best = False
		if f1 > best_f1:
			best_top1 = test_top1
			best_f1=f1
			best_fn=fn
			best_precision=precision
			best_tn=tn
			best_fp=fp
			best_recall=recall
			best_accuracy=accuracy1
			is_best = True
		logging.info('Saving models......')
		save_checkpoint({
			'epoch': epoch,
			'net': net.state_dict(),
			'prec@1': test_top1,
			'F1':f1
		}, is_best, args.save_root)

	print(f"TP: {best_tp}")
	print(f"FP: {best_fp}")
	print(f"TN: {best_tn}")
	print(f"FN: {best_fn}")
	print(f"Recall: {best_recall:.4f}")
	print(f"Precision: {best_precision:.4f}")
	print(f"Accuracy: {best_accuracy:.4f}")
	print(f"F1: {best_f1:.4f}")


def train(train_loader, net, optimizer, criterion, epoch):
	batch_time = AverageMeter()
	data_time  = AverageMeter()
	losses     = AverageMeter()
	top1       = AverageMeter()

	net.train()
	all_targets = []
	all_preds = []
	end = time.time()
	for i, (img, target) in enumerate(train_loader, start=1):
		data_time.update(time.time() - end)

		# if args.cuda:
		# 	img = img.cuda(non_blocking=True)
		# 	target = target.cuda(non_blocking=True)

		stem, rb1, rb2, rb3, feat, out = net(img)
		loss = criterion(out, target)

		preds = out.argmax(dim=1).cpu().numpy()
		all_preds.extend(preds)
		all_targets.extend(target.cpu().numpy())

		prec1, = accuracy(out, target, topk=(1,))
		losses.update(loss.item(), img.size(0))
		top1.update(prec1.item(), img.size(0))


		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			f1_epoch = f1_score(all_targets, all_preds, average='macro')
			log_str = ('Epoch[{0}]:[{1:03}/{2:03}] '
					   'Time:{batch_time.val:.4f} '
					   'Data:{data_time.val:.4f}  '
					   'loss:{losses.val:.4f}({losses.avg:.4f})  '
					   'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
					   'F1-macro:{f1:.4f}'.format(
				epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
				losses=losses, top1=top1,  f1=f1_epoch))
			logging.info(log_str)



def test(test_loader, net, criterion):
	losses = AverageMeter()
	top1   = AverageMeter()

	all_targets = []
	all_preds = []

	net.eval()

	end = time.time()
	for i, (img, target) in enumerate(test_loader, start=1):
		# if args.cuda:
		# 	img = img.cuda(non_blocking=True)
		# 	target = target.cuda(non_blocking=True)

		with torch.no_grad():
			stem, rb1, rb2, rb3, feat, out = net(img)
			loss = criterion(out, target)

		prec1,= accuracy(out, target, topk=(1,))
		losses.update(loss.item(), img.size(0))
		top1.update(prec1.item(), img.size(0))

		preds = out.argmax(dim=1).cpu().numpy()
		all_preds.extend(preds)
		all_targets.extend(target.cpu().numpy())

	tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
	precision = precision_score(all_targets, all_preds)
	recall = recall_score(all_targets, all_preds)
	accuracy1 = accuracy_score(all_targets, all_preds)
	f1 = f1_score(all_targets, all_preds)
	f_l = [losses.avg, top1.avg, f1]

	logging.info('Loss: {:.4f}, Prec@1: {:.2f}, F1-macro: {:.4f}'.format(*f_l))

	return top1.avg,f1,precision,recall,accuracy1,tp,fp,tn,fn


def adjust_lr(optimizer, epoch):
	scale   = 0.1
	lr_list =  [args.lr] * 100
	lr_list += [args.lr*scale] * 50
	lr_list += [args.lr*scale*scale] * 50

	lr = lr_list[epoch-1]
	logging.info('Epoch: {}  lr: {:.3f}'.format(epoch, lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


if __name__ == '__main__':
	main()