import torch
from torch import nn
import pytorch_lightning as pl
from long_transformers import RobertaLongForMaskedLM
from transformers import RobertaConfig
# from transformers import RobertaClassificationHead

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
		# x = torch.sigmoid(x)
        return x

class relevantClassifier(pl.LightningModule):
	
	def __init__(self, args):
		super().__init__()
		self.lr = args['lr']
		self.BERT_encoder = RobertaLongForMaskedLM.from_pretrained(args['pretrained_model'], output_hidden_states=True, gradient_checkpointing=True)
		self.binaryClassifier = RobertaClassificationHead(self.BERT_encoder.config)

		# self.init_weights()
		'''
			self.binaryClassifier = torch.nn.Sequential(
										torch.nn.Linear(768, 300), 
										torch.nn.ReLU(), 
										torch.nn.Linear(300, 50), 
										torch.nn.ReLU(), 
										torch.nn.Linear(50, 1), 
										torch.nn.Sigmoid()
									)
		'''
		self.loss = torch.nn.BCEWithLogitsLoss()
		self.save_hyperparameters()
	
	def forward(self, batch):
		input_ids, attention_mask = batch
		encoder_outputs = self.BERT_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)[1][-1] # last hidden state
		outputs = self.binaryClassifier(encoder_outputs)
		return outputs
	
	def training_step(self, batch, batch_idx):
		input_ids, attention_mask, label = batch
		input_ids = torch.squeeze(input_ids)
		attention_mask = torch.squeeze(attention_mask)
		label = torch.squeeze(label, dim=0)
		outputs = self.forward((input_ids, attention_mask))
		# loss = torch.nn.functional.binary_cross_entropy(outputs, label)
		loss = self.loss(outputs, label)
		pred = [1 if output > 0.5 else 0 for output in outputs]
		succ_pred = sum([1 if p == r else 0 for p, r in zip(pred, label)])
		# self.log('train_loss', loss, prog_bar=True)
		self.log('acc', succ_pred / len(label), prog_bar=True)
		return loss
	
	def validation_step(self, batch, batch_idx):
		input_ids, attention_mask, label = batch
		input_ids = torch.squeeze(input_ids)
		attention_mask = torch.squeeze(attention_mask)
		label = torch.squeeze(label, dim=0)
		outputs = self.forward((input_ids, attention_mask))
		# loss = torch.nn.functional.binary_cross_entropy(outputs, label)
		loss = self.loss(outputs, label)
		pred = [1 if output > 0.5 else 0 for output in outputs]
		succ_pred = sum([1 if p == r else 0 for p, r in zip(pred, label)])
		# self.log('eval_loss', loss, prog_bar=True)
		self.log('acc', succ_pred / len(label), prog_bar=True)
		return loss
	
	def test_step(self, batch, batch_idx):
		input_ids, attention_mask, doc_id = batch
		logits = self.forward((input_ids, attention_mask))
		scores = torch.sigmoid(logits)
		all_pred = [(score, doc_id) for score, doc_id in zip(scores, doc_id)]
		doc_score, doc_cnt = {}, {}
		for score, doc_id in all_pred:
			if doc_id not in doc_score:
				doc_score[doc_id] = 0
				doc_cnt[doc_id] = 0
			doc_score[doc_id] += score.item()
			doc_cnt[doc_id] += 1

		avg_scores = [(k, v / doc_cnt[k]) for k, v in doc_score.items()]
		relevance_order = [k for k, v in sorted(avg_scores, key=lambda score: score[1], reverse=True)]
		return {'order': 0.25}
		return {'order': relevance_order}

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
		return optimizer
