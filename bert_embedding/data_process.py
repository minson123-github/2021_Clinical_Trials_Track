import os
import copy
import json
import torch
import random
import pickle
from tqdm import tqdm
from tokenizers import AddedToken
from transformers import BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ProcessPoolExecutor

# GLOBAL_MAX_POS = 4096

class trainDataset(Dataset):
	def __init__(self, anchor, positive, negative):
		self.anchor = torch.LongTensor(anchor)
		self.positive = torch.LongTensor(positive)
		self.negative = torch.LongTensor(negative)
	
	def __len__(self):
		return len(self.anchor)

	def __getitem__(self, idx):
		return self.anchor[idx], self.positive[idx], self.negative[idx]

class embeddingDataset(Dataset):
	def __init__(self, input_ids, doc_id):
		self.input_ids = torch.LongTensor(input_ids)
		self.doc_id = doc_id
	
	def __len__(self):
		return len(self.input_ids)
	
	def __getitem__(self, idx):
		return self.input_ids[idx], self.doc_id[idx]

def get_file_ids(args):
	file_list = os.listdir(args['file_content'])
	file_ids = []
	for filename in tqdm(file_list, position=0, leave=False, desc='Get file ids'):
		file_path = os.path.join(args['file_content'], filename)
		with open(file_path, 'r') as fp:
			file_contents = fp.readlines()
		for i in range(0, len(file_contents), 2):
			file_id, file_content = file_contents[i][:-1], file_contents[i + 1][:-1]
			file_ids.append(file_id)
	return file_ids

def request_file_content(args, file_id_list, save_path):
	file_id_set = set(file_id_list)
	file_list = os.listdir(args['file_content'])
	request_contents = []
	for filename in tqdm(file_list, position=1, leave=False, desc='Get content'):
		file_path = os.path.join(args['file_content'], filename)
		with open(file_path, 'r') as fp:
			file_contents = fp.readlines()
		for i in range(0, len(file_contents), 2):
			file_id, file_content = file_contents[i][:-1], file_contents[i + 1][:-1]
			if file_id in file_id_set:
				request_contents.append(file_content)
	
	with open(save_path, 'w') as fp:
		json.dump(request_contents, fp)

def request_multi_file_content(args, file_ids_lists, save_paths):
	file_id_sets = [set(id_list) for id_list in file_ids_lists]
	file_list = os.listdir(args['file_content'])
	request_contents = [None] * len(save_paths)
	for i in range(len(save_paths)):
		request_contents[i] = []
	
	for filename in tqdm(file_list, position=0, desc='Get content'):
		file_path = os.path.join(args['file_content'], filename)
		with open(file_path, 'r') as fp:
			file_contents = fp.readlines()
		for i in range(0, len(file_contents), 2):
			file_id, file_content = file_contents[i][:-1], file_contents[i + 1][:-1]
			for pid in range(len(save_paths)):
				if file_id in file_id_sets[pid]:
					request_contents[pid].append(file_content)
	
	for pid in range(len(save_paths)):
		with open(save_paths[pid], 'w') as fp:
			json.dump(request_contents[pid], fp)

def sample_non_relevant(args, relevant_ids, file_ids):
	relevant_id_set = set(relevant_ids)
	target_size = int(len(relevant_ids) * args['sample_ratio'])
	sample_index = [idx for idx in range(len(file_ids))]
	non_relevant_ids = []
	n_non_relevant = 0
	random.shuffle(sample_index)
	for idx in sample_index:
		sample_file_id = file_ids[idx]
		if n_non_relevant < target_size:
			if sample_file_id not in relevant_id_set:
				non_relevant_ids.append(sample_file_id)
				n_non_relevant += 1
		else:
			break
	return non_relevant_ids

def sample_from_elastic(args, relevant_ids, elastic_ids):
	relevant_id_set = set(relevant_ids)
	target_size = int(len(relevant_ids) * args['sample_ratio'])
	non_relevant_ids = []
	for fid in elastic_ids:
		if fid not in relevant_id_set and target_size > 0:
			target_size -= 1
			non_relevant_ids.append(fid)
	return non_relevant_ids

def refresh_dir(_dir):
	if not os.path.exists(_dir):
		os.mkdir(_dir)
	files = os.listdir(_dir)
	for filename in files:
		file_path = os.path.join(_dir, filename)
		os.remove(file_path)

def check_dir(_dir, n_query):
	if not os.path.exists(_dir):
		return False
	files = os.listdir(_dir)
	if len(files) < n_query:
		return False
	for i in range(n_query):
		file_path = os.path.join(_dir, 'part_{}.json'.format(i))
		if not os.path.exists(file_path):
			return False
	return True

def embedding_dataset_tokenize(tokenizer, file_path, worker_id, part_id):
	with open(file_path, 'r') as fp:
		file_contents = fp.readlines()
	
	results = {'doc_id': [], 'input_ids': []}
	for i in tqdm(range(0, len(file_contents), 2), leave=False, position=worker_id + 1, desc='Worker-{}'.format(worker_id)):
		file_id, file_content = file_contents[i][:-1], file_contents[i + 1][:-1]
		doc_tokenize = tokenizer(
						file_content, 
						padding='max_length', 
						truncation=True,
						stride=128, 
						return_tensors='np', 
						return_overflowing_tokens=True
					).input_ids

		for input_ids in doc_tokenize:
			results['doc_id'].append(file_id)
			results['input_ids'].append(input_ids.tolist())
	
	with open('embedding_tokenize/part_{}.json'.format(part_id), 'w') as fp:
		json.dump(results, fp)

def train_dataset_tokenize(tokenizer, query_terms, worker_id, part_id):
	with open('relevant/part_{}.json'.format(part_id), 'r') as fp:
		relevant = json.load(fp)
	with open('non-relevant/part_{}.json'.format(part_id), 'r') as fp:
		non_relevant = json.load(fp)
	
	results = {'positive': [], 'negative': []}

	query_input_ids = tokenizer(
						query_terms, 
						padding='max_length', 
						truncation=True,
						stride=64, 
						return_tensors='np', 
						return_overflowing_tokens=True
					).input_ids

	for para in tqdm(relevant, position=worker_id, leave=False, desc='relevant'):
		para_input_ids = tokenizer(
							para, 
							padding='max_length', 
							truncation=True,
							stride=64, 
							return_tensors='np', 
							return_overflowing_tokens=True
						).input_ids

		for input_ids in para_input_ids:
			results['positive'].append(input_ids.tolist())
	
	for para in tqdm(non_relevant, position=worker_id, leave=False, desc='non-relevant'):
		para_input_ids = tokenizer(
							para, 
							padding='max_length', 
							truncation=True, 
							stride=64, 
							return_tensors='np', 
							return_overflowing_tokens=True
						).input_ids

		for input_ids in para_input_ids:
			results['negative'].append(input_ids.tolist())
	
	sub_data = {'anchor': [], 'positive': [], 'negative': []}
	for input_ids in query_input_ids:
		for positive, negative in zip(results['positive'], results['negative']):
			sub_data['anchor'].append(input_ids.tolist())
			sub_data['positive'].append(positive)
			sub_data['negative'].append(negative)
	
	with open('tokenize/part_{}.json'.format(part_id), 'w') as fp:
		json.dump(sub_data, fp)

def train_collate_fn(batch):
	batch_anchor, batch_positive, batch_negative = zip(*batch)
	return torch.stack(batch_anchor), torch.stack(batch_positive), torch.stack(batch_negative)

def embedding_collate_fn(batch):
	batch_input_ids, batch_doc_ids = zip(*batch)
	return torch.stack(batch_input_ids), batch_doc_ids

def get_elastic_dataloader(args):
	random.seed(args['seed'])
	query_terms_list = os.listdir(args['query_parsed'])
	query_terms_list = sorted(query_terms_list, key=lambda filename: int(filename))
	query_terms = []
	for query_file in query_terms_list:
		file_path = os.path.join(args['query_parsed'], query_file)
		with open(file_path, 'r') as fp:
			query_terms.append(fp.readline())
	
	relevance_ids = []
	with open(args['relevance_id'], 'r') as fp:
		relevance_info = fp.readlines()
		for info in relevance_info:
			relevance_ids.append(info[:-1].split(' '))
	
	elastic_ids = []
	with open(args['elastic_result'], 'r') as fp:
		elastic_results = fp.readlines()
		for result in elastic_results:
			elastic_ids.append(result.split(' ')[:10000])
	
	n_query = len(query_terms_list)
	if not check_dir('tokenize', n_query):
		if not check_dir('relevant', n_query) or not check_dir('non-relevant', n_query):
			# file_ids = get_file_ids(args)
			refresh_dir('relevant')
			refresh_dir('non-relevant')
			non_relevance_ids = []
			relevance_paths, non_relevance_paths = [], []
			for i in tqdm(range(n_query), position=0, leave=False, desc='write-relevance'):
				non_relevance = sample_from_elastic(args, relevance_ids[i], elastic_ids[i])
				non_relevance_ids.append(non_relevance)
				relevance_paths.append('relevant/part_{}.json'.format(i))
				non_relevance_paths.append('non-relevant/part_{}.json'.format(i))
			request_multi_file_content(args, relevance_ids, relevance_paths)
			request_multi_file_content(args, non_relevance_ids, non_relevance_paths)
		
		tokenizer = BertTokenizerFast.from_pretrained(args['pretrained_model'], model_max_length=512)

		refresh_dir('tokenize')
		part_idx = 0
		with ProcessPoolExecutor(max_workers=args['n_worker']) as executor:
			while part_idx < n_query:
				futures = []
				for worker_id in range(args['n_worker']):
					future = executor.submit(train_dataset_tokenize, tokenizer, query_terms[part_idx], worker_id, part_idx)
					part_idx += 1
					futures.append(future)
					if part_idx == n_query:
						break
				for future in futures:
					future.result()
	
	if not os.path.exists('train_dataset.pickle'):
		tokenize = {'anchor':[], 'positive':[], 'negative':[]}
		for i in tqdm(range(n_query), position=0, leave=False, desc='combine'):
			with open('tokenize/part_{}.json'.format(i), 'r') as fp:
				part = json.load(fp)
			for anchor, positive, negative in zip(part['anchor'], part['positive'], part['negative']):
				if random.random() <= args['train_dataset_ratio']:
					tokenize['anchor'].append(anchor)
					tokenize['positive'].append(positive)
					tokenize['negative'].append(negative)
			print(len(tokenize['positive']), len(tokenize['positive'][0]))
			# break
		train_dataset = trainDataset(
						tokenize['anchor'], 
						tokenize['positive'], 
						tokenize['negative']
					)
		with open('train_dataset.pickle', 'wb') as fp:
			pickle.dump(train_dataset, fp)
	else:
		with open('train_dataset.pickle', 'rb') as fp:
			train_dataset = pickle.load(fp)
	
	train_dataloader = DataLoader(
						train_dataset, 
						batch_size=args['batch_size'], 
						shuffle=True, 
						num_workers=4 * args['n_gpu'], 
						pin_memory=True, 
						persistent_workers=True, 
						collate_fn=train_collate_fn
					)
	return train_dataloader

def get_train_dataloader(args):
	random.seed(args['seed'])
	query_terms_list = os.listdir(args['query_parsed'])
	query_terms_list = sorted(query_terms_list, key=lambda filename: int(filename))
	query_terms = []
	for query_file in query_terms_list:
		file_path = os.path.join(args['query_parsed'], query_file)
		with open(file_path, 'r') as fp:
			query_terms.append(fp.readline()[:-1])
	
	relevance_ids = []
	with open(args['relevance_id'], 'r') as fp:
		relevance_info = fp.readlines()
		for info in relevance_info:
			relevance_ids.append(info[:-1].split(' '))
	
	n_query = len(query_terms_list)
	if not check_dir('tokenize', n_query):
		if not check_dir('relevant', n_query) or not check_dir('non-relevant', n_query):
			file_ids = get_file_ids(args)
			refresh_dir('relevant')
			refresh_dir('non-relevant')
			for i in tqdm(range(n_query), position=0, leave=False, desc='write-relevance'):
				non_relevance_ids = sample_non_relevant(args, relevance_ids[i], file_ids)
				request_file_content(args, relevance_ids[i], 'relevant/part_{}.json'.format(i))
				request_file_content(args, non_relevance_ids, 'non-relevant/part_{}.json'.format(i))
		
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

		refresh_dir('tokenize')
		part_idx = 0
		with ProcessPoolExecutor(max_workers=args['n_worker']) as executor:
			while part_idx < n_query:
				futures = []
				for worker_id in range(args['n_worker']):
					future = executor.submit(train_dataset_tokenize, tokenizer, query_terms[part_idx], worker_id, part_idx)
					part_idx += 1
					futures.append(future)
					if part_idx == n_query:
						break
				for future in futures:
					future.result()
	
	if not os.path.exists('train_dataset.pickle'):
		tokenize = {'input_ids':[], 'query_input_ids':[], 'target':[]}
		for i in tqdm(range(n_query), position=0, leave=False, desc='combine'):
			with open('tokenize/part_{}.json'.format(i), 'r') as fp:
				part = json.load(fp)
			for input_ids, query_input_ids, target in zip(part['input_ids'], part['query_input_ids'], part['target']):
				if random.random() <= args['train_dataset_ratio']:
					tokenize['input_ids'].append(input_ids)
					tokenize['query_input_ids'].append(query_input_ids)
					tokenize['target'].append(target)
			# break
		train_dataset = trainDataset(
						tokenize['input_ids'], 
						tokenize['query_input_ids'], 
						tokenize['target']
					)
		with open('train_dataset.pickle', 'wb') as fp:
			pickle.dump(train_dataset, fp)
	else:
		with open('train_dataset.pickle', 'rb') as fp:
			train_dataset = pickle.load(fp)
	
	train_dataloader = DataLoader(
						train_dataset, 
						batch_size=args['batch_size'], 
						shuffle=True, 
						num_workers=4 * args['n_gpu'], 
						pin_memory=True, 
						persistent_workers=True, 
						collate_fn=train_collate_fn
					)
	return train_dataloader

def get_embedding_dataset(args, pid):
	file_list = os.listdir(args['file_content'])
	file_list = sorted(file_list, key=lambda filename: int(filename))
	print('Number of file part: {}'.format(len(file_list)), flush=True)
	tokenizer = BertTokenizerFast.from_pretrained(args['pretrained_model'], model_max_length=512)

	if not check_dir('embedding_tokenize', len(file_list)):
		refresh_dir('embedding_tokenize')
		part_id = 0
		with ProcessPoolExecutor(max_workers=args['n_worker']) as executor:
			for i in tqdm(range(0, len(file_list), args['n_worker']), position=0, leave=False, desc='tokenize'):
				batch_files = file_list[i: min(i + args['n_worker'], len(file_list))]
				futures = []
				for worker_id, filename in enumerate(batch_files):
					file_path = os.path.join(args['file_content'], filename)
					future = executor.submit(embedding_dataset_tokenize, tokenizer, file_path, worker_id, part_id)
					part_id += 1
					futures.append(future)
			
				for future in futures:
					future.result()

	tokenize = {'doc_id': [], 'input_ids': []}
	if pid == "query":	
		query_terms_list = os.listdir(args['query_parsed'])
		query_terms_list = sorted(query_terms_list, key=lambda filename: int(filename))

		for query_file in query_terms_list:
			file_path = os.path.join(args['query_parsed'], query_file)
			with open(file_path, 'r') as fp:
				query_term = fp.readline()[:-1]

			query_input_ids = tokenizer(
								query_term, 
								padding='max_length', 
								truncation=True, 
								stride=128, 
								return_tensors='np', 
								return_overflowing_tokens=True
							).input_ids

			for input_ids in query_input_ids:
				tokenize['doc_id'].append(query_file)
				tokenize['input_ids'].append(input_ids.tolist())
	else:
		with open('embedding_tokenize/part_{}.json'.format(pid), 'r') as fp:
			tokenize = json.load(fp)	
	
	embedding_dataset = embeddingDataset(
							tokenize['input_ids'], 
							tokenize['doc_id']
						)

	return embedding_dataset

def get_partial_dataset(args, pid):
	tokenizer = BertTokenizerFast.from_pretrained(args['pretrained_model'], model_max_length=512)

	tokenize = {'doc_id': [], 'input_ids': []}
	if pid == 'query':
		with open('sample/query/1', 'r') as fp:
			query_term = fp.readline()

		query_input_ids = tokenizer(
							query_term, 
							padding='max_length', 
							truncation=True, 
							stride=64, 
							return_tensors='np', 
							return_overflowing_tokens=True
						).input_ids

		for input_ids in query_input_ids:
			tokenize['doc_id'].append('1')
			tokenize['input_ids'].append(input_ids.tolist())
	else:
		if os.path.exists('embedding_tokenize/doc.pickle'):
			with open('embedding_tokenize/doc.pickle', 'rb') as fp:
				tokenize = pickle.load(fp)
		else:
			with open('sample/data/1', 'r') as fp:
				file_contents = fp.readlines()
			for i in tqdm(range(0, len(file_contents), 2), position=0, leave=False, desc='tokenize'):
				file_id, file_cont = file_contents[i][:-1], file_contents[i + 1][:-1]
				file_input_ids = tokenizer(
									file_cont, 
									padding='max_length', 
									truncation=True, 
									stride=64, 
									return_tensors='np', 
									return_overflowing_tokens=True
								).input_ids
				for input_ids in file_input_ids:
					tokenize['doc_id'].append(file_id)
					tokenize['input_ids'].append(input_ids.tolist())
			with open('embedding_tokenize/doc.pickle', 'wb') as fp:
				pickle.dump(tokenize, fp)

	embedding_dataset = embeddingDataset(
							tokenize['input_ids'], 
							tokenize['doc_id']
						)

	return embedding_dataset
