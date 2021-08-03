import os
import torch
from tqdm import tqdm
from config import get_args
from data_process import get_train_dataloader, get_test_dataloader
from model import relevantClassifier
from pytorch_lightning import Trainer, seed_everything, Callback
import pytorch_lightning as pl
from fairscale.nn import checkpoint_wrapper, auto_wrap, wrap
# import torch.multiprocessing

class CheckpointEveryNSteps(Callback):
	def __init__(self, save_freq, save_dir):
		self.save_dir = save_dir
		self.save_freq = save_freq
		self.save_cnt = 0
	
	def on_batch_end(self, trainer: Trainer, _):
		global_step = trainer.global_step
		if (global_step + 1) % self.save_freq == 0:
			self.save_cnt += 1
			filename = 'checkpoint_{}.ckpt'.format(self.save_cnt)
			ckpt_path = os.path.join(self.save_dir, filename)
			trainer.save_checkpoint(ckpt_path)

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
	
	model = relevantClassifier(args)

	train_dataloader, eval_dataloader = get_train_dataloader(args)
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
	lr_finder = trainer.tuner.lr_find(model, train_dataloaders=train_dataloader)
	print(lr_finder.results)
	new_lr = lr_finder.suggestion()
	model.hparams.lr = new_lr
	'''

	print('The learning rate be used for training is: ', model.lr, flush=True)

	print('start to training model.', flush=True)
	if args['eval_mode']:
		trainer.fit(model, train_dataloader, eval_dataloader)
	else:
		trainer.fit(model, train_dataloader)

def test(args):
	# torch.multiprocessing.set_sharing_strategy('file_system')
	n_query = len(os.listdir(args['query_terms']))
	
	model_save_path = os.path.join(args['model_dir'], 'checkpoints', 'checkpoint_5.ckpt')
	model = relevantClassifier.load_from_checkpoint(model_save_path)
	print(isinstance(model, pl.LightningModule))
	'''
	device_ids = [gpu_id for gpu_id in range(args['n_gpu'])]
	model = torch.nn.DataParallel(model, device_ids)
	model.to('cuda:0')

	predict_results = []

	model.eval()
	with torch.no_grad():
		for qid in tqdm(range(n_query), position=0, desc='all-query'):
			test_dataloader = get_test_dataloader(args, qid)
			doc_scores, doc_cnt = {}, {}
			for (input_ids, attention_mask, doc_ids) in tqdm(test_dataloader, position=1, desc='predict', leave=False):
				logits = model((input_ids.cuda(), attention_mask.cuda()))
				relevant_scores = torch.sigmoid(logits)
				for score, doc_id in zip(relevant_scores, doc_ids):
					if doc_id not in doc_scores:
						doc_scores[doc_id] = 0
						doc_cnt[doc_id] = 0
					doc_scores[doc_id] += score.item()
					doc_cnt[doc_id] += 1
			doc_scores = [k for k, v in sorted(doc_scores.items(), key=lambda item: item[1] / doc_cnt[item[0]], reverse=True)]
			predict_results.append(doc_scores)
	'''
	
	trainer = Trainer(
					deterministic=True, 
					gpus=args['n_gpu'], 
					num_nodes=1, 
					#accelerator="ddp", 
					#plugins="ddp_sharded", 
				)

	predict_results = []
	for qid in range(n_query):
		test_dataloader = get_test_dataloader(args, qid)
		doc_scores = trainer.test(model, test_dataloader)
		print(doc_scores, flush=True)
		predict_results.append(doc_scores)
	
	print('start to saving predict results.', flush=True)
	with open(args['predict_file'], 'w') as fp:
		for qid, pred in enumerate(predict_results):
			for doc_id in pred:
				fp.write('{} {}\n'.format(qid + 1, doc_id))
	print('write predict results finished.', flush=True)

if __name__ == '__main__':
	args = get_args()
	if args['mode'] == 'train':
		train(args)
	if args['mode'] == 'test':
		test(args)
