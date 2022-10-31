import os
import json
import argparse
from tqdm import tqdm

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print("GPU device:", torch.cuda.device_count())  # Check how many CUDA capable devices you have, 2)

unifiedqa_models = {
    "small": "allenai/unifiedqa-t5-small",  # 231M
    "base": "allenai/unifiedqa-t5-base",  # 850M
    "large": "allenai/unifiedqa-t5-large",  # 2.7G
    "3b": "allenai/unifiedqa-t5-3b",  # 10.6G
    "11b": "allenai/unifiedqa-t5-11b",  # 42.1G
}

option_inds = ["A", "B", "C", "D", "E", "F"]


class Solver(object):

    def __init__(self, args):

        self.verbose = args.verbose

        # build the model
        model_name = unifiedqa_models[args.model]  # you can specify the model size here, 42.1G
        print("# model:", model_name)
        print("# Loading the model ...")

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

        if args.check_point != None:
            model_path = os.path.join(args.saved_models, args.label, args.check_point)
            print(f"# Loading the check point: {model_path}")
            self.model.load_state_dict(torch.load(model_path))

        self.model.eval()

        self.device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")  # one GPU
        self.model.to(self.device)

        # generation arguments
        self.num_beams = args.num_beams
        self.max_input_length = args.max_input_length
        self.truncation = True if self.max_input_length != None else False
        self.max_output_length = args.max_output_length

        # load the data
        data_file = os.path.join(args.data_root, f'problems_{args.test_split}.json')
        print(f"# Loading data from {data_file}...")
        self.problems = json.load(open(data_file))

        self.pids = list(self.problems.keys())
        if args.test_num > 0:
            self.pids = self.pids[:min(len(self.pids), args.test_num)]
        print("# number of problems for %s:" % (args.test_split), len(self.pids), "\n")

    def generate_answer(self, input_string):
        input_ids = self.tokenizer.encode(input_string,
                                          return_tensors="pt",
                                          truncation=self.truncation,
                                          max_length=self.max_input_length).to(self.device)

        tokens = self.model.generate(input_ids,
                                     num_beams=self.num_beams,
                                     min_length=1,
                                     max_length=self.max_output_length,
                                     early_stopping=True)

        outputs = self.tokenizer.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return outputs[0].strip()

    def get_input_text(self, pid):
        # table
        table_title = self.problems[pid]['table_title']
        table = self.problems[pid]['table']
        if table_title:
            table = table_title + "\n" + table

        # question
        question = self.problems[pid]['question']
        unit = self.problems[pid]['unit']
        choices = self.problems[pid]['choices']

        if unit:
            question = question + f" (Unit: {unit})"

        if choices:
            for i, c in enumerate(choices):
                question += f" ({option_inds[i]}) {c}"

        # final input
        text = table + "\n" + question
        text = text.replace("\n", " \\n ").strip()

        return text

    def get_label_answer(self, pid):
        return self.problems[pid]['answer']

    def get_pred_answer(self, pid):
        text = self.get_input_text(pid)
        # generate the predicrion answer string
        prediction = self.generate_answer(text)

        if self.verbose:
            print(f"# [Input]\n{text}")
            print(f"# [Prediction]\n{prediction}")

        return prediction


if __name__ == '__main__':

    ## Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data/tabmwp')
    parser.add_argument('--output', type=str, default='../results/t5')
    parser.add_argument('--saved_models', type=str, default='../saved_models/t5')
    parser.add_argument('--check_point', type=str, default=None)

    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'test1k'])
    parser.add_argument('--test_num', type=int, default=-1)
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--label', type=str, default='exp0')

    parser.add_argument('--model', type=str, default='small')
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--max_input_length', type=int, default=None)
    parser.add_argument('--max_output_length', type=int, default=100)

    args = parser.parse_args()

    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    ## Test
    solver = Solver(args)

    correct = 0
    results = {}

    for pid in tqdm(solver.pids):
        results[pid] = {}

        output = solver.get_pred_answer(pid)
        answer = solver.get_label_answer(pid)

        results[pid]["output"] = output
        results[pid]["answer"] = answer

    ## Save results
    data = {}
    data['args'] = vars(args)
    data['results'] = results

    os.makedirs(f"{args.output}", exist_ok=True)

    result_file = "{}/{}_{}.json".format(args.output, args.label, args.model)
    print("\n# Saved results to", result_file)

    with open(result_file, 'w') as f:
        json.dump(data, f, indent=2, separators=(',', ': '))
