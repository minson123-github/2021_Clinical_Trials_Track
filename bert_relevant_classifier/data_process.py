from torch.utils.data import Dataset, DataLoader
import json
from typing import List
import os
import random
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import torch
from transformers import RobertaTokenizerFast

GLOBAL_MAX_POS = 4096

class paragraphDataset(Dataset):
	
	def __init__(self, input_ids, attention_mask, is_relevant=None):
		self.input_ids = input_ids
		self.attention_mask = attention_mask
		self.label = is_relevant
	
	def __len__(self):
		return len(self.input_ids)
	
	def __getitem__(self, idx):
		if self.label == None:
			return self.input_ids[idx], self.attention_mask[idx]
		return self.input_ids[idx], self.attention_mask[idx], self.label[idx]

def clear_dir(remove_dir):
	if os.path.exists(remove_dir):
		remove_files = os.listdir(remove_dir)
		for remove_file in remove_files:
			remove_path = os.path.join(remove_dir, remove_file)
			os.remove(remove_path)

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
		tokenizer = RobertaTokenizerFast.from_pretrained(args['pretrained_model'], model_max_length=GLOBAL_MAX_POS)

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
	
# def get_test_dataloader(args):
