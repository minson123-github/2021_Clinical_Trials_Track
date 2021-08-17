import os
import sys
import torch
import pickle
from tqdm import tqdm
from config import get_config
from data_process import get_elastic_dataloader, get_embedding_dataset, refresh_dir, embedding_collate_fn, get_partial_dataset
from pytorch_lightning import Trainer, seed_everything, Callback
from model import embeddingNet
from pytorch_lightning.plugins.training_type import DDPShardedPlugin
from pytorch_lightning.plugins import DeepSpeedPlugin
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from inferenceSampler import SequentialDistributedSampler
import torch.multiprocessing as mp

os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["MASTER_ADDR"] = "140.112.31.65"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "25487"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

class CheckpointEveryNSteps(Callback):
	def __init__(self, save_freq, save_dir):
		self.save_dir = save_dir
		self.save_freq = save_freq
		self.save_cnt = 0
	
	def on_batch_end(self, trainer: Trainer, _):
		global_step = trainer.global_step
		if (global_step + 1) % self.save_freq == 0:
			self.save_cnt += 1
			filename = '0checkpoint_{}.ckpt'.format(self.save_cnt)
			ckpt_path = os.path.join(self.save_dir, filename)
			trainer.save_checkpoint(ckpt_path)

def normalize(v):
	v_length = sum([x * x for x in v]) ** 0.5
	v_hat = [x / v_length for x in v]
	return v_hat

def sum_vec(vec1, vec2):
	return [x + y for x, y in zip(vec1, vec2)]

def sub_embedding(rank, offset, ckpt_path, world_size, is_half, batch_size):
	torch.distributed.init_process_group('nccl', rank=offset + rank, world_size=world_size)
	print('initial rank {} process'.format(torch.distributed.get_rank()), flush=True)
	# dist.barrier()
	torch.cuda.set_device(rank)
	device = torch.device('cuda', rank)

	model = embeddingNet.load_from_checkpoint(ckpt_path)
	#if is_half:
	#model.half()
	model.to(device)
	# print('Setting DDP...', flush=True)
	model = DDP(model, device_ids=[rank], output_device=rank)
	# print('Setting finished...', flush=True)
	model.eval()
	
	with open('shared/dataset.pickle', 'rb') as fp:
		test_dataset = pickle.load(fp)
	test_sampler = SequentialDistributedSampler(test_dataset, batch_size=batch_size)
	test_dataloader = torch.utils.data.DataLoader(
						test_dataset, 
						batch_size=batch_size, 
						sampler=test_sampler, 
						pin_memory=True, 
						num_workers=4, 
						collate_fn=embedding_collate_fn
					)
	
	avg_emb, n_seq = {}, {}

	predicts = []
	with torch.no_grad():
		for input_ids, doc_ids in tqdm(test_dataloader, position=0, leave=False, desc='GPU-{}'.format(rank)):
			preds = model(input_ids.cuda())
			embeddings = [[x.item() for x in pred] for pred in preds]
			for emb, doc_id in zip(embeddings, doc_ids):
				unit_emb = normalize(emb)
				predicts.append((unit_emb, doc_id))
			dist.barrier()
	
	with open('shared/pred_{}.pickle'.format(offset + rank), 'wb') as fp:
		pickle.dump(predicts, fp)

def combine_predict(dataset_length):
	pred_files = []
	for filename in os.listdir('shared'):
		if filename[0:5] == 'pred_':
			pred_files.append(filename)

	pred_files = sorted(pred_files, key=lambda filename: int(filename[5:-7])) # sort by process rank
	full_pred = []
	for filename in tqdm(pred_files, position=0, leave=False, desc='combine'):
		file_path = os.path.join('shared', filename)
		with open(file_path, 'rb') as fp:
			part_pred = pickle.load(fp)
		for p in part_pred:
			full_pred.append(p)
	
	full_pred = full_pred[: dataset_length]
	
	avg_emb, n_seq = {}, {}
	for emb, doc_id in full_pred:
		if doc_id not in avg_emb:
			avg_emb[doc_id] = emb
			n_seq[doc_id] = 1
		else:
			avg_emb[doc_id] = sum_vec(avg_emb[doc_id], emb)
			n_seq[doc_id] += 1
	
	result = {}
	for k, v in avg_emb.items():
		d = n_seq[k]
		result[k] = [x / d for x in v]
	return result

def partial_embedding(args):
	ckpt_path = 'embedding_model.ckpt'
	if not os.path.exists('embedding'):
		os.makedirs('embedding')
	if not os.path.exists('embedding/query.pickle'):
		print('start to process query embedding', flush=True)
		if args['gpu_offset'] == 0:
			query_dataset = get_partial_dataset(args, 'query')
			dataset_length = len(query_dataset)
			print('Length of dataset: {}'.format(dataset_length))
			refresh_dir('shared')
			with open('shared/dataset.pickle', 'wb') as fp:
				pickle.dump(query_dataset, fp)
		mp.spawn(
					sub_embedding, 
					args=(args['gpu_offset'], ckpt_path, args['n_gpu'], False, args['batch_size']), 
					nprocs=args['node_gpus'], 
					join=True
				)
		if args['gpu_offset'] == 0:
			predict = combine_predict(dataset_length)
			with open('embedding/query.pickle', 'wb') as fp:
				pickle.dump(predict, fp)
	
	if not os.path.exists('embedding/doc.pickle'):
		print('start to process document embedding', flush=True)
		if args['gpu_offset'] == 0:
			query_dataset = get_partial_dataset(args, 'doc')
			dataset_length = len(query_dataset)
			print('Length of dataset: {}'.format(dataset_length))
			refresh_dir('shared')
			with open('shared/dataset.pickle', 'wb') as fp:
				pickle.dump(query_dataset, fp)
		mp.spawn(
					sub_embedding, 
					args=(args['gpu_offset'], ckpt_path, args['n_gpu'], False, args['batch_size']), 
					nprocs=args['node_gpus'], 
					join=True
				)
		if args['gpu_offset'] == 0:
			predict = combine_predict(dataset_length)
			with open('embedding/doc.pickle', 'wb') as fp:
				pickle.dump(predict, fp)

def embedding(args):
	ckpt_path = 'embedding_model.ckpt'
	if not os.path.exists('embedding'):
		os.makedirs('embedding')
	if not os.path.exists('embedding/query.pickle'):
		print('start to process query embedding', flush=True)
		if args['gpu_offset'] == 0:
			query_dataset = get_embedding_dataset(args, 'query')
			dataset_length = len(query_dataset)
			print('Length of dataset: {}'.format(dataset_length))
			refresh_dir('shared')
			with open('shared/dataset.pickle', 'wb') as fp:
				pickle.dump(query_dataset, fp)
		mp.spawn(
					sub_embedding, 
					args=(args['gpu_offset'], ckpt_path, args['n_gpu'], False, args['batch_size']), 
					nprocs=args['node_gpus'], 
					join=True
				)
		if args['gpu_offset'] == 0:
			predict = combine_predict(dataset_length)
			with open('embedding/query.pickle', 'wb') as fp:
				pickle.dump(predict, fp)
	
	n_part = len(os.listdir('embedding_tokenize'))
	for pid in range(n_part):
		if os.path.exists('embedding/part_{}.pickle'.format(pid)):
			continue
		print('start to process part {}/{} file'.format(pid + 1, n_part), flush=True)
		if args['gpu_offset'] == 0:
			part_dataset = get_embedding_dataset(args, pid)
			dataset_length = len(part_dataset)
			print('Length of dataset: {}'.format(dataset_length))
			refresh_dir('shared')
			with open('shared/dataset.pickle', 'wb') as fp:
				pickle.dump(part_dataset, fp)
		mp.spawn(
					sub_embedding, 
					args=(args['gpu_offset'] , ckpt_path, args['n_gpu'], False, args['batch_size']), 
					nprocs=args['node_gpus'], 
					join=True
				)
		if args['gpu_offset'] == 0:
			predict = combine_predict(dataset_length)
			with open('embedding/part_{}.pickle'.format(pid), 'wb') as fp:
				pickle.dump(predict, fp)

def train(args):
	seed_everything(args['seed'], workers=True)
	
	save_path = os.path.join(args['model_dir'], 'model')
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	
	callbacks = []
	if type(args['saving_steps']) != type(None):
		callbacks = [CheckpointEveryNSteps(args['saving_steps'], os.path.join(args['model_dir'], 'checkpoints'))]
		if not os.path.exists(os.path.join(args['model_dir'], 'checkpoints')):
			os.makedirs(os.path.join(args['model_dir'], 'checkpoints'))
	
	model = embeddingNet(args['pretrained_model'], args['lr'])
	#model = embeddingNet.load_from_checkpoint('embedding_model.ckpt')
	train_dataloader = get_elastic_dataloader(args)

	trainer = Trainer(
					deterministic=True, 
					gpus=args['n_gpu'], 
					default_root_dir=save_path, 
					max_epochs=args['n_epoch'], 
					num_nodes=1, 
					# precision=16, 
					accelerator="ddp", 
					# amp_backend='apex', 
					# plugins='deepspeed_stage_2', 
					plugins='ddp_sharded', 
					# accumulate_grad_batches=16, 
					# plugins=DeepSpeedPlugin(deepspeed_config), 
					callbacks = callbacks, 
				)

	'''
	lr_finder = trainer.tuner.lr_find(model, train_dataloader)
	print(lr_finder.results)
	new_lr = lr_finder.suggestion()
	model.hparams.lr = new_lr
	'''
	print('The learning rate be used for training is: ', model.lr, flush=True)
	
	# model.hparams.lr = args['lr']
	trainer.fit(model, train_dataloader)

if __name__ == '__main__':
	gpu_status = torch.cuda.is_available()
	print('GPU Status: ', gpu_status)
	config = get_config()
	if config['mode'] == 'train':
		train(config)
	if config['mode'] == 'embedding':
		embedding(config)
	if config['mode'] == 'partial':
		partial_embedding(config)
