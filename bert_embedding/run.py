import os
import torch
from config import get_config
from data_process import get_train_dataloader
from pytorch_lightning import Trainer, seed_everything, Callback
from model import embeddingNet

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
	
	model = embeddingNet(args['pretrained_model'], args['lr'])
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
	if config['mode'] == 'predict':
		pass
