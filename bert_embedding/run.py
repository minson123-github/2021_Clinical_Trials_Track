import os
import sys
import torch
import pickle
from tqdm import tqdm
from config import get_config
from data_process import get_train_dataloader, get_embedding_dataloader, refresh_dir
from pytorch_lightning import Trainer, seed_everything, Callback
from model import embeddingNet
from pytorch_lightning.plugins.training_type import DDPShardedPlugin

os.environ["TOKENIZERS_PARALLELISM"] = "true"

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

def part_embedding(args, trainer, model, pid):
	embedding_dataloader = get_embedding_dataloader(args, pid)
	with torch.no_grad():
		embedding = trainer.predict(model, embedding_dataloader)
	'''
		model.to('cuda')
		model.eval()
		embedding = []
		with torch.no_grad():
			for input_ids, doc_id in tqdm(embedding_dataloader, position=0, desc='predict'):
				emb = model(input_ids.cuda()).cpu().detach().numpy().tolist()
				embedding.append((emb, doc_id))
	'''

	'''
	if model.global_rank != 0:
		sys.exit(0)
	'''
	
	doc_emb, doc_cnt = {}, {}

	for batch_embedding, batch_doc_ids in embedding:
		for emb, doc_id in zip(batch_embedding, batch_doc_ids):
			unit_emb = normalize(emb)
			if doc_id not in doc_emb:
				doc_emb[doc_id] = unit_emb
				doc_cnt[doc_id] = 1
			else:
				doc_emb[doc_id] = sum_vec(doc_emb[doc_id], unit_emb)
				doc_cnt[doc_id] += 1
	
	result = {}
	for k, v in doc_emb.items():
		avg_emb = [x / doc_cnt[k] for x in v]
		result[k] = avg_emb
	
	return result

def embedding(args):
	model = embeddingNet.load_from_checkpoint(os.path.join(args['model_dir'], 'checkpoints', 'checkpoint1_2.ckpt'))
	trainer = Trainer(
					deterministic=True, 
					gpus=args['n_gpu'], 
					num_nodes=1, 
					accelerator="ddp", 
					plugins=DDPShardedPlugin(), 
				)
	if not os.path.exists('embedding'):
		os.makedirs('embedding')
	if not os.path.exists('embedding/query.pickle'):
		query_embedding = part_embedding(args, trainer, model, "query")
		if model.global_rank == 0:
			with open('embedding/query.pickle', 'wb') as fp:
				pickle.dump(query_embedding, fp)
	
	n_part = len(os.listdir(args['file_content']))
	for pid in range(n_part):
		if not os.path.exists('embedding/part_{}.pickle'.format(pid)):
			print('start to embedding part {}/{}'.format(pid + 1, n_part), flush=True)
			_part = part_embedding(args, trainer, model, pid)
			if model.global_rank == 0:
				with open('embedding/part_{}.pickle'.format(pid), 'wb') as fp:
					pickle.dump(_part, fp)

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
