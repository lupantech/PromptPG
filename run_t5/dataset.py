import os
import json
import torch


def get_input_text(problem, option_inds):
    # table
    table_title = problem['table_title']
    table = problem['table']
    if table_title:
        table = table_title + "\n" + table

    # question
    question = problem['question']
    unit = problem['unit']
    choices = problem['choices']

    if unit:
        question = question + f" (Unit: {unit})"

    if choices:
        for i, c in enumerate(choices):
            question += f" ({option_inds[i]}) {c}"

    # final input
    text = table + "\n" + question
    text = text.replace("\n", " \\n ").strip()

    return text


def _load_dataset(data_root, split, option_inds):
    """
    Load the dataset.
    """
    # load the data entries/annotations
    problems = json.load(open(os.path.join(data_root, f'problems_{split}.json')))
    print("number of problems for %s:" % (split), len(problems))

    pids = list(problems.keys())
    print("number of problems for %s:" % (split), len(pids))

    entries = []
    for pid in pids:
        prob = {}
        prob['pid'] = pid
        # prob['problem'] = problems[pid]
        prob['input_text'] = get_input_text(problems[pid], option_inds)
        prob['answer'] = problems[pid]['answer']
        entries.append(prob)

    return entries


## Create PyTorch dataset
class TMQADataset(torch.utils.data.Dataset):

    def __init__(self, data_root, data_split, tokenizer, args):
        self.data_root = data_root
        self.data_split = data_split
        self.tokenizer = tokenizer
        self.option_inds = args.option_inds
        self.max_input_length = args.max_input_length
        self.max_output_length = args.max_output_length

        # load the data entries/annotations
        self.entries = _load_dataset(data_root, data_split, self.option_inds)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        # get one entry of data
        entry = self.entries[idx]  # annotation

        pid = entry['pid']
        input_text = entry['input_text']
        answer = entry['answer']

        # print("# input_string:", text)
        # print("# output_string:", answer)

        batch = {"input_strings": input_text, "output_strings": answer}

        return pid, batch
