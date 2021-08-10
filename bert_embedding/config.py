import argparse

def get_config():
	parser = argparse.ArgumentParser()
	# parser.add_argument('--full_dataset', type=int, default=0, help='Whether to use full dataset for training.')
	parser.add_argument('--train_dataset_ratio', type=float, help='The ratio of dataset be used in training.')
	parser.add_argument('--file_content', type=str, help='The directory contains file content.')
	parser.add_argument('--query_parsed', type=str, help='The query terms parsed by amazon comprehend.')
	parser.add_argument('--relevance_id', type=str, help='The file contains file ids. One line for one query.')
	parser.add_argument('--sample_ratio', type=float, help='The ratio of sample non-relevant files.')
	parser.add_argument('--seed', type=int, help='Random seed.')
	parser.add_argument('--n_worker', type=int, default=1, help='Number of workers for tokenize.')
	parser.add_argument('--model_dir', type=str, help='The directory for saving model and checkpoints.')
	parser.add_argument('--saving_steps', type=int, help='Number of training steps to save checkpoint.')
	parser.add_argument('--pretrained_model', type=str, default='simonlevine/bioclinical-roberta-long', help='The pretrained model for fine-tuning.')
	parser.add_argument('--n_gpu', type=int, default=1, help='Number of GPU be used in training.')
	parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for training.')
	parser.add_argument('--n_epoch', type=int, default=1, help='Number of epoch for training.')
	# parser.add_argument('--update_freq', type=int, help='Frequency of update fixed model.')
	parser.add_argument('--mode', type=str, help='Mode: train or embedding')
	parser.add_argument('--batch_size', type=int, help='Batch size for training.')
	args = parser.parse_args()
	return vars(args)
