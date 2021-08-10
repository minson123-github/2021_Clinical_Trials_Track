import torch
import pytorch_lightning as pl
from long_transformers import RobertaLongForMaskedLM

class embeddingNet(pl.LightningModule):
	
	def __init__(self, pretrained_model, lr):
		super().__init__()
		self.emb_net = RobertaLongForMaskedLM.from_pretrained(pretrained_model, output_hidden_states=True, gradient_checkpointing=True)
		self.lr = lr
		self.save_hyperparameters()
		self.cosine_loss = torch.nn.CosineEmbeddingLoss()
	
	def forward(self, input_ids):
		emb = self.emb_net(input_ids=input_ids, output_hidden_states=True)[1][-1][:, 0, :]
		return emb
	
	def training_step(self, batch, batch_idx):
		input_ids, query_input_ids, target = batch
		para_emb = self.forward(input_ids)
		query_emb = self.forward(query_input_ids)
		loss = self.cosine_loss(para_emb, query_emb, target)
		return loss
	
	def validation_step(self, batch, batch_idx):
		input_ids, query_input_ids, target = batch
		para_emb = self.forward(input_ids)
		query_emb = self.forward(query_input_ids)
		loss = self.cosine_loss(para_emb, query_emb, target)
		return loss
	
	def predict_step(self, batch, batch_idx):
		input_ids, doc_id = batch
		preds = self.forward(input_ids)
		emb = [[x.item() for x in pred] for pred in preds]
		return emb, doc_id
	
	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
		return optimizer
