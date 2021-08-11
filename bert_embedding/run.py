import os
import sys
import torch
import pickle
from tqdm import tqdm
from config import get_config
from data_process import get_train_dataloader, get_embedding_dataset, refresh_dir, embedding_collate_fn
from pytorch_lightning import Trainer, seed_everything, Callback
from model import embeddingNet
from pytorch_lightning.plugins.training_type import DDPShardedPlugin
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from inferenceSampler import SequentialDistributedSampler

os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"

class CheckpointEveryNSteps(Callback):
	def __init__(self, save_freq, save_dir):
		self.save_dir = save_dir
		self.save_freq = save_freq
		self.save_cnt = 0
	
	def on_batch_end(self, trainer: Trainer, _):
		global_step = trainer.global_step
		if (global_step + 1) % self.save_freq == 0:
			self.save_cnt += 1
			filename = 'checkpoint1_{}.ckpt'.format(self.save_cnt)
			ckpt_path = os.path.join(self.save_dir, filename)
			trainer.save_checkpoint(ckpt_path)

def normalize(v):
	v_length = sum([x * x for x in v]) ** 0.5
	v_hat = [x / v_length for x in v]
	return v_hat

def sum_vec(vec1, vec2):
	return [x + y for x, y in zip(vec1, vec2)]

def sub_embedding(rank, ckpt_path, world_size, is_half, batch_size):
	dist.init_process_group('nccl', rank=rank, world_size=world_size)
	torch.cuda.set_device(rank)
	device = torch.device('cuda', rank)

	model = embeddingNet.load_from_checkpoint(ckpt_path)
	if is_half:
		model.half()
	model.to(device)
	model = DDP(model, device_ids=[rank], output_device=rank)
	
	with open('shared/dataset.pickle', 'rb') as fp:
		test_dataset = pickle.load(fp)
	test_sampler = SequentialDistributedSampler(test_dataset, batch_size=batch_size)
	test_dataloader = torch.utils.data.DataLoader(
						test_dataset, 
						batch_size=batch_size, 
						sampler=test_sampler, 
						pin_memory=True, 
						num_worker=4, 
						collate_fn=embedding_collate_fn
					)
	
	avg_emb, n_seq = {}, {}

	predicts = []
	with torch.no_grad():
		for input_ids, doc_ids in tqdm(test_dataloader, position=rank + 1, leave=False, desc='GPU-{}'.format(rank)):
			preds = model(input_ids.cuda())
			embeddings = [[x.item() for x in pred] for pred in preds]
			for emb, doc_id in zip(embeddings, doc_ids):
				unit_emb = normalize(emb)
				predicts.append((unit_emb, doc_id))
	
	with open('shared/pred_{}.pickle'.format(rank), 'wb') as fp:
		pickle.dump(predicts, fp)

def combine_predict():
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

def embedding(args):
	ckpt_path = 'embedding_model.ckpt'
	if not os.path.exists('embedding'):
		os.makedirs('embedding')
	if not os.path.exists('embedding/query'):
		query_dataset = get_embedding_dataset(args, 'query')
		refresh_dir('shared')
		mp.spawn(
					sub_embedding, 
					args=(ckpt_path, args['n_gpu'], False, args['batch_size']), 
					nproc=args['n_gpu'], 
					join=True
				)
		predict = combine_predict()
		with open('embedding/query.pickle', 'wb') as fp:
			pickle.dump(predict, fp)
	
	n_part = os.listdir('embedding_tokenize')
	for pid in tqdm(range(n_part), leave=False, position=0, 'total'):
		part_dataset = get_embedding_dataset(args, pid)
		refresh_dir('shared')
		mp.spawn(
					sub_embedding, 
					args=(ckpt_path, args['n_gpu'], False, args['batch_size']), 
					nproc=args['n_gpu'], 
					join=True
				)
		predict = combine_predict()
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
	
	# model = embeddingNet(args['pretrained_model'], args['lr'])
	model = embeddingNet.load_from_checkpoint(os.path.join(args['model_dir'], 'checkpoints', 'checkpoint_2.ckpt'))
	train_dataloader = get_train_dataloader(args)

	trainer = Trainer(
					deterministic=True, 
					gpus=args['n_gpu'], 
					default_root_dir=save_path, 
					max_epochs=args['n_epoch'], 
					num_nodes=1, 
					# precision=16, 
					accelerator="ddp", 
					plugins="ddp_sharded", 
					callbacks = callbacks, 
				)

	'''
	lr_finder = trainer.tuner.lr_find(model, train_dataloader)
	print(lr_finder.results)
	new_lr = lr_finder.suggestion()
	model.hparams.lr = new_lr
	'''
	print('The learning rate be used for training is: ', model.lr, flush=True)
	trainer.fit(model, train_dataloader)

if __name__ == '__main__':
	config = get_config()
	if config['mode'] == 'train':
		train(config)
	if config['mode'] == 'embedding':
		embedding(config)
