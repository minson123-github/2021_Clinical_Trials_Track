import torch
import pytorch_lightning as pl
from transformers import BertForMaskedLM
'''
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.runtime.fp16.onebit.adam import OnebitAdam
from fairscale.optim import OSS
'''

class embeddingNet(pl.LightningModule):
	
	def __init__(self, pretrained_model, lr):
		super().__init__()
		self.embedding = BertForMaskedLM.from_pretrained(pretrained_model, output_hidden_states=True, gradient_checkpointing=True)
		self.lr = lr
		self.save_hyperparameters()
		self.TripletLoss = torch.nn.TripletMarginWithDistanceLoss(distance_function=torch.nn.CosineSimilarity(), margin=0.6)
	
	def forward(self, input_ids):
		emb = self.embedding(input_ids=input_ids, output_hidden_states=True).hidden_states[-1][:, 0, :]
		return emb
	
	def training_step(self, batch, batch_idx):
		anchor, positive, negative = batch
		anchor_emb = self.forward(anchor)
		positive_emb = self.forward(positive)
		negative_emb = self.forward(negative)
		loss = self.TripletLoss(anchor_emb, negative_emb, positive_emb)
		return loss
	
	def validation_step(self, batch, batch_idx):
		anchor, positive, negative = batch
		anchor_emb = self.forward(anchor)
		positive_emb = self.forward(positive)
		negative_emb = self.forward(negative)
		loss = self.TripletLoss(anchor_emb, negative_emb, positive_emb)
		return loss
	
	def predict_step(self, batch, batch_idx):
		input_ids, doc_id = batch
		preds = self.forward(input_ids)
		emb = [[x.item() for x in pred] for pred in preds]
		return emb, doc_id

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
		#return OnebitAdam(self.parameters(), lr=self.lr)
		# optimizer = OSS(self.parameters(), optim=torch.optim.AdamW, lr=self.lr)
		return optimizer
