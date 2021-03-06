from torch.utils.data import Dataset, DataLoader
import json
from typing import List
import os
import random
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import torch
from transformers import RobertaTokenizerFast
from tokenizers import AddedToken

GLOBAL_MAX_POS = 4096

class paragraphDataset(Dataset):
	
	def __init__(self, input_ids, attention_mask, is_relevant):
		self.input_ids = input_ids
		self.attention_mask = attention_mask
		self.label = is_relevant
	
	def __len__(self):
		return len(self.input_ids)
	
	def __getitem__(self, idx):
		return self.input_ids[idx], self.attention_mask[idx], self.label[idx]

class predictDataset(Dataset):
	
	def __init__(self, input_ids, attention_mask, doc_id):
		self.input_ids = input_ids
		self.attention_mask = attention_mask
		self.doc_id = doc_id
	
	def __len__(self):
		return len(self.input_ids)

	def __getitem__(self, idx):
		return self.input_ids[idx], self.attention_mask[idx], self.doc_id[idx]

def clear_dir(remove_dir):
	if os.path.exists(remove_dir):
		remove_files = os.listdir(remove_dir)
		for remove_file in remove_files:
			remove_path = os.path.join(remove_dir, remove_file)
			os.remove(remove_path)

def tokenize_predict_paragraph(tokenizer, worker_id, query_terms):

	### Load data via file ##########################################
	with open('predict/part_{}.json'.format(worker_id), 'r') as fp:
		predict_paragraph = json.load(fp)
	#################################################################
	tokenize_results = {'input_ids': [], 'attention_mask': [], 'doc_id': []}
	for para, doc_id in tqdm(predict_paragraph, position=worker_id + 1, leave=False):
		para_tokenize = tokenizer(
					query_terms, 
					para,
					padding='max_length',
					truncation='only_second',
					stride=256, 
					return_tensors='np', 
					return_attention_mask=True, 
					return_overflowing_tokens=True)
		
		for tokens in para_tokenize['input_ids']:
			tokenize_results['input_ids'].append(tokens.tolist())
			tokenize_results['doc_id'].append(doc_id)

		for tokens in para_tokenize['attention_mask']:
			tokenize_results['attention_mask'].append(tokens.tolist())
	
	with open('predict_tokenize/part_{}.json'.format(worker_id), 'w') as fp:
		json.dump(tokenize_results, fp)

def tokenize_paragraph(tokenizer, worker_id, part_id):

	### Load data via file ##########################################
	with open('query_terms/part_{}.json'.format(part_id), 'r') as fp:
		query_terms = json.load(fp)
	
	with open('relevant/part_{}.json'.format(part_id), 'r') as fp:
		relevant = json.load(fp)
	
	with open('non_relevant/part_{}.json'.format(part_id), 'r') as fp:
		non_relevant = json.load(fp)
	#################################################################

	tokenize_results = {'input_ids': [], 'attention_mask': [], 'relevant': []}

	for para in tqdm(relevant, position=worker_id, leave=False, desc='relevant'):
		para_tokenize = tokenizer(
					query_terms, 
					para,
					padding='max_length',
					truncation='only_second',
					stride=256, 
					return_tensors='np', 
					return_attention_mask=True, 
					return_overflowing_tokens=True)
		
		for tokens in para_tokenize['input_ids']:
			tokenize_results['input_ids'].append(tokens.tolist())
			tokenize_results['relevant'].append(1)
		
		for tokens in para_tokenize['attention_mask']:
			tokenize_results['attention_mask'].append(tokens.tolist())
	
	for para in tqdm(non_relevant, position=worker_id, leave=False, desc='non-relevant'):
		para_tokenize = tokenizer(
					query_terms, 
					para,
					padding='max_length',
					truncation='only_second',
					stride=256, 
					return_tensors='np', 
					return_attention_mask=True, 
					return_overflowing_tokens=True)

		
		for tokens in para_tokenize['input_ids']:
			tokenize_results['input_ids'].append(tokens.tolist())
			tokenize_results['relevant'].append(0)
		
		for tokens in para_tokenize['attention_mask']:
			tokenize_results['attention_mask'].append(tokens.tolist())
	
	#### saving tokenize results into json file ##################
	with open('tokenize/part_{}.json'.format(part_id), 'w') as fp:
		json.dump(tokenize_results, fp)
	##############################################################

def get_relevant(args):
	relevant_fid = []

	with open(args['relevant_fid'], 'r') as fp:
		contents = fp.readlines()
		for content in contents:
			fids = content[:-1].split(' ')
			relevant_fid.append(fids)
	return relevant_fid

def get_all_fid(args):
	file_list = os.listdir(args['file_content'])
	fids = []
	for file_name in file_list:
		file_path = os.path.join(args['file_content'], file_name)

		with open(file_path, 'r') as fp:
			contents = fp.readlines()
			for i, content in enumerate(contents):
				if i % 2 == 0:
					fids.append(content[:-1])
	return fids

def write_to_file(args, fid_list, save_path):
	fid_set = set(fid_list)
	file_list = os.listdir(args['file_content'])
	
	file_contents = []
	for file_name in file_list:
		file_path = os.path.join(args['file_content'], file_name)

		with open(file_path, 'r') as fp:
			contents = fp.readlines()
			for i in range(0, len(contents), 2):
				fid, content = contents[i][:-1], contents[i + 1][:-1]
				if fid in fid_set:
					file_contents.append(content)
	
	with open(save_path, 'w') as fp:
		json.dump(file_contents, fp)

def write_to_predict_file(args, fid_list, save_path):
	fid_set = [set(worker_list) for worker_list in fid_list]
	file_list = os.listdir(args['file_content'])
	
	file_contents = [None] * len(save_path)
	for i in range(len(save_path)):
		file_contents[i] = []
	for file_name in tqdm(file_list, position=1, leave=False, desc='write-content'):
		file_path = os.path.join(args['file_content'], file_name)

		with open(file_path, 'r') as fp:
			contents = fp.readlines()
			for i in range(0, len(contents), 2):
				fid, content = contents[i][:-1], contents[i + 1][:-1]
				for worker_id in range(len(save_path)):
					if fid in fid_set[worker_id]:
						file_contents[worker_id].append((content, fid))
	
	for worker_id in range(len(save_path)):
		with open(save_path[worker_id], 'w') as fp:
			json.dump(file_contents[worker_id], fp)

def check_utility_info(_dir):
	if not os.path.exists(_dir):
		return False
	n_parts = len(os.listdir(_dir))
	if n_parts < 30:
		return False
	for i in range(n_parts):
		if not os.path.exists(os.path.join(_dir, 'part_{}.json'.format(i))):
			return False
	return True

def refresh_dir(_dir):
	if not os.path.exists(_dir):
		os.mkdir(_dir)
	clear_dir(_dir)

def train_collate_fn(train_batch):
	# batch_input_ids, batch_attention_mask, batch_relevant = train_batch
	batch_input_ids = [x for x, _, __ in train_batch]
	batch_attention_mask = [x for _, x, __ in train_batch]
	batch_relevant = [[x] for _, __, x in train_batch]
	# for input_ids, attention_mask, relevant in train_batch:
		# batch_input_ids.append(torch.LongTensor(input_ids).unsqueeze(0))
		# batch_attention_mask.append(torch.FloatTensor(attention_mask).unsqueeze(0))
		# batch_relevant.append(torch.FloatTensor([relevant]).unsqueeze(0))
	
	# batch_input_ids = torch.cat(batch_input_ids, dim=0)
	# batch_attention_mask = torch.cat(batch_attention_mask, dim=0)
	# batch_relevant = torch.cat(batch_relevant, dim=0)
	batch_input_ids = torch.LongTensor(batch_input_ids)
	batch_attention_mask = torch.FloatTensor(batch_attention_mask)
	batch_relevant = torch.FloatTensor(batch_relevant)

	return batch_input_ids, batch_attention_mask, batch_relevant

def test_collate_fn(test_batch):
	batch_input_ids, batch_attention_mask, batch_doc_ids = [], [], []
	for input_ids, attention_mask, doc_id in test_batch:
		batch_input_ids.append(input_ids)
		batch_attention_mask.append(attention_mask)
		batch_doc_ids.append(doc_id)

	batch_input_ids = torch.LongTensor(batch_input_ids)
	batch_attention_mask = torch.FloatTensor(batch_attention_mask)
	return batch_input_ids, batch_attention_mask, batch_doc_ids

def get_train_dataloader(args):
	random.seed(args['seed']) # fix random seed

	relevant_fid = get_relevant(args)
	all_fid = get_all_fid(args)
	access_order = [i for i in range(len(all_fid))]
	
	### generate non-relevant fids and write file contents for tokenize ###
	if not check_utility_info('tokenize'):
		refresh_dir('tokenize')
		if not check_utility_info('relevant') or not check_utility_info('non_relevant') or not check_utility_info('query_terms'):
			refresh_dir('relevant')
			refresh_dir('non_relevant')
			refresh_dir('query_terms')
			print('Generate non-relevant samples and write file contents for tokenize.', flush=True)
			for i, relevant in enumerate(tqdm(relevant_fid, position=0)):
				random.shuffle(access_order) # shuffle fid access order
				relevant_set = set(relevant)
				non_relevant = []
				non_relevant_size = 0
				target_size = min(int(args['mult_non_relevant'] * len(relevant)), len(access_order))
				# pick up non-relevant files randomly
				for access_index in access_order:
					if non_relevant_size < target_size and all_fid[access_index] not in relevant_set:
						non_relevant.append(all_fid[access_index])
						non_relevant_size += 1

				write_to_file(args, relevant, 'relevant/part_{}.json'.format(i))
				write_to_file(args, non_relevant, 'non_relevant/part_{}.json'.format(i))
				# read query terms
				with open(os.path.join(args['query_terms'], str(i + 1)), 'r') as fp:
					query_terms = fp.readline()[:-1]
				with open('query_terms/part_{}.json'.format(i), 'w') as fp:
					json.dump(query_terms, fp)
		
		print('Start to tokenize paragraph data parallelly.', flush=True)
		part_idx = 0
		tokenizer = RobertaTokenizerFast.from_pretrained(
						args['pretrained_model'], 
						model_max_length=GLOBAL_MAX_POS, 
						bos_token='<s>', 
						eos_token='</s>', 
						unk_token='<unk>', 
						sep_token='</s>', 
						pad_token='<pad>', 
						cls_token='<s>', 
						mask_token=AddedToken("<mask>", rstrip=False, lstrip=True, single_word=False, normalized=False)
					)

		with ProcessPoolExecutor(max_workers=args['n_worker']) as executor:
			while part_idx < len(relevant_fid):
				futures = []
				for i in range(args['n_worker']):
					if part_idx < len(relevant_fid):
						futures.append(executor.submit(tokenize_paragraph, tokenizer, i, part_idx))
						part_idx += 1
					else:
						break
				for future in futures:
					future.result()
	
	all_tokenize = {'input_ids': [], 'attention_mask': [], 'relevant': []}
	for i in tqdm(range(len(relevant_fid))):
		with open('tokenize/part_{}.json'.format(i), 'r') as fp:
			tokenize_results = json.load(fp)
			for input_ids, attention_mask, relevant in zip(tokenize_results['input_ids'], tokenize_results['attention_mask'], tokenize_results['relevant']):
				all_tokenize['input_ids'].append(input_ids)
				all_tokenize['attention_mask'].append(attention_mask)
				all_tokenize['relevant'].append([relevant])
	
	print('pre-split data into several batches.', flush=True)
	data_size = len(all_tokenize['input_ids'])
	batch_data = {'input_ids': [], 'attention_mask': [], 'relevant': []}
	for i in tqdm(range(0, len(all_tokenize['input_ids']), args['batch_size'])):
		if i + args['batch_size'] >= data_size:
			break
		batch_input_ids = all_tokenize['input_ids'][i: min(i + args['batch_size'], data_size)]
		batch_attention_mask = all_tokenize['attention_mask'][i: min(i + args['batch_size'], data_size)]
		batch_relevant = all_tokenize['relevant'][i: min(i + args['batch_size'], data_size)]
		batch_data['input_ids'].append(batch_input_ids)
		batch_data['attention_mask'].append(batch_attention_mask)
		batch_data['relevant'].append(batch_relevant)
	
	all_tokenize = batch_data
	
	all_tokenize['input_ids'] = torch.LongTensor(all_tokenize['input_ids'])
	all_tokenize['attention_mask'] = torch.FloatTensor(all_tokenize['attention_mask'])
	all_tokenize['relevant'] = torch.FloatTensor(all_tokenize['relevant'])
	# print(all_tokenize['input_ids'].size())
	# print(all_tokenize['attention_mask'].size())
	# print(all_tokenize['relevant'].size())
	
	if args['eval_mode']:
		dataset_size = len(all_tokenize['relevant'])
		train_dataset = paragraphDataset(
						all_tokenize['input_ids'][: int(dataset_size * (1 - args['eval_ratio']))],
						all_tokenize['attention_mask'][: int(dataset_size * (1 - args['eval_ratio']))],
						all_tokenize['relevant'][: int(dataset_size * (1 - args['eval_ratio']))])
		eval_dataset = paragraphDataset(
						all_tokenize['input_ids'][int(dataset_size * (1 - args['eval_ratio'])): ],
						all_tokenize['attention_mask'][int(dataset_size * (1 - args['eval_ratio'])): ],
						all_tokenize['relevant'][int(dataset_size * (1 - args['eval_ratio'])): ])
		train_dataloader = DataLoader(
								train_dataset, 
								# batch_size=args['batch_size'], 
								batch_size=1, 
								shuffle=True, 
								num_workers=4 * args['n_gpu'], 
								pin_memory=True, 
								persistent_workers=True, 
								# collate_fn=train_collate_fn
							)
		eval_dataloader = DataLoader(
								eval_dataset, 
								# batch_size=args['batch_size'], 
								batch_size=1, 
								shuffle=False, 
								num_workers=4 * args['n_gpu'], 
								pin_memory=True, 
								persistent_workers=True, 
								# collate_fn=train_collate_fn
							)
		return train_dataloader, eval_dataloader
	else:
		train_dataset = paragraphDataset(
							all_tokenize['input_ids'], 
							all_tokenize['attention_mask'], 
							all_tokenize['relevant']
						)

		train_dataloader = DataLoader(
							train_dataset, 
							# batch_size=args['batch_size'], 
							batch_size=1, 
							shuffle=True, 
							num_workers=4 * args['n_gpu'], 
							pin_memory=True, 
							persistent_workers=True, 
							# prefetch_factor=4, 
							# collate_fn=train_collate_fn
						)
		return train_dataloader, None
	
def get_test_dataloader(args, i):
	# get i-th(0~29) query predict file id dataloader
	with open(args['predict_relevance'], 'r') as fp:
		predict_fids = fp.readlines()[i][:-1].split(' ')
		with open(os.path.join(args['query_terms'], str(i + 1)), 'r') as fp:
			query_terms = fp.readline()[:-1]
	
	refresh_dir('predict')
	refresh_dir('predict_tokenize')
	distri_workers = [None] * args['n_worker']
	for idx, fid in enumerate(predict_fids[: 80]):
		worker_id = idx % args['n_worker']
		if type(distri_workers[worker_id]) == type(None):
			distri_workers[worker_id] = []
		distri_workers[worker_id].append(fid)
	tokenizer = RobertaTokenizerFast.from_pretrained(
					args['pretrained_model'], 
					model_max_length=GLOBAL_MAX_POS, 
					bos_token='<s>', 
					eos_token='</s>', 
					unk_token='<unk>', 
					sep_token='</s>', 
					pad_token='<pad>', 
					cls_token='<s>', 
					mask_token=AddedToken("<mask>", rstrip=False, lstrip=True, single_word=False, normalized=False)
				)

	save_path = ['predict/part_{}.json'.format(worker_id) for worker_id in range(args['n_worker'])]
	write_to_predict_file(args, distri_workers, save_path)
	
	
	with ProcessPoolExecutor(max_workers=args['n_worker']) as executor:
		futures = [executor.submit(tokenize_predict_paragraph, tokenizer, worker_id, query_terms) for worker_id in range(args['n_worker'])]
		for future in futures:
			future.result()

	tokenize_results = {'input_ids': [], 'attention_mask': [], 'doc_id': []}
	for worker_id in tqdm(range(args['n_worker']), position=1, leave=False, desc='combine'):
		with open('predict_tokenize/part_{}.json'.format(worker_id), 'r') as fp:
			part_tokenize = json.load(fp)	
		for input_ids in part_tokenize['input_ids']:
			tokenize_results['input_ids'].append(input_ids)
		for attention_mask in part_tokenize['attention_mask']:
			tokenize_results['attention_mask'].append(attention_mask)
		for doc_id in part_tokenize['doc_id']:
			tokenize_results['doc_id'].append(doc_id)
	
	test_dataset = predictDataset(
						tokenize_results['input_ids'][: 80], 
						tokenize_results['attention_mask'][: 80], 
						tokenize_results['doc_id'][: 80]
					)
	test_dataloader = DataLoader(
						test_dataset, 
						batch_size=args['batch_size'], 
						shuffle=False, 
						num_workers=4 * args['n_gpu'], 
						pin_memory=True, 
						persistent_workers=True, 
						collate_fn=test_collate_fn
					)
	return test_dataloader
