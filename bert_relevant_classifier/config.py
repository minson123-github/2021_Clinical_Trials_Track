import argparse

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--file_content', type=str, help='The directory with file contents.')
	parser.add_argument('--seed', type=int, help='Random seed.')
	parser.add_argument('--relevant_fid', type=str, default='ans.txt', help='File contain relevant file id.')
	parser.add_argument('--query_terms', type=str, help='The directory with query terms.')
	parser.add_argument('--n_worker', type=int, help='Number of worker for tokenize.')
	parser.add_argument('--batch_size', type=int, help='Batch size for training phase.')
	parser.add_argument('--model_dir', type=str, default='ckpt', help='Directory for saving model.')
	parser.add_argument('--pretrained_model', type=str, default='simonlevine/bioclinical-roberta-long', help='The pretrained model for fine-tuning.')
	parser.add_argument('--mult_non_relevant', type=float, default=1, help='The multiple of relevant file size which decides non-relevant file size.')
	parser.add_argument('--eval_mode', type=int, default=0, help='Whether to split evaluate dataset for evaluating model performance.')
	parser.add_argument('--eval_ratio', type=float, default=0, help='The ratio of evaluate dataset in all dataset.')
	parser.add_argument('--n_gpu', type=int, default=1, help='Number of GPU be used in training.')
	parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training.')
	parser.add_argument('--n_epoch', type=int, default=1, help='Number of epoch for training.')
	parser.add_argument('--saving_steps', type=int, help='Number of step for saving model.')
	parser.add_argument('--mode', type=str, help='Mode: train or test')
	args = parser.parse_args()
	return vars(args)
