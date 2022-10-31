import os
import json
import argparse
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tapex_models = {
    "tapex-base": "microsoft/tapex-base-finetuned-wtq",
    "tapex-large": "microsoft/tapex-large-finetuned-wtq"
}

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print("GPU device:", torch.cuda.device_count())  # Check how many CUDA capable devices you have, 2)

option_inds = ["A", "B", "C", "D", "E", "F"]


class Solver(object):

    def __init__(self, args):

        self.verbose = args.verbose
        self.max_length = args.max_length

        # build the model
        model_name = tapex_models[args.model]  # you can specify the model size here
        print("# model:", model_name)
        print("# Loading the model ...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if args.check_point != None:
            model_path = os.path.join(args.saved_models, args.label, args.check_point)
            print(f"# Loading the check point: {model_path}")
            self.model.load_state_dict(torch.load(model_path))

        self.model.eval()

        self.device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")  # one GPU
        self.model.to(self.device)

        # load the data
        data_file = os.path.join(args.data_root, f'problems_{args.test_split}.json')
        print(f"# Loading data from {data_file}...")
        self.problems = json.load(open(data_file))

        self.pids = list(self.problems.keys())
        if args.test_num > 0:
            self.pids = self.pids[:min(len(self.pids), args.test_num)]
        print("# number of problems for %s:" % (args.test_split), len(self.pids), "\n")

    def generate_answer(self, pd_table, input_string):
        # QA
        inputs = self.tokenizer(pd_table, input_string, return_tensors="pt")
        inputs = inputs.to(self.device)

        # let the model generate an answer autoregressively
        outputs = self.model.generate(**inputs, max_length=self.max_length)

        # decode back to text
        predicted_answer = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        return predicted_answer.strip()

    def get_input_text(self, pid):
        # table
        table_for_pd = self.problems[pid]['table_for_pd']
        pd_table = pd.DataFrame.from_dict(table_for_pd)

        # question
        question = self.problems[pid]['question']
        unit = self.problems[pid]['unit']
        choices = self.problems[pid]['choices']
        if unit:
            question = question + f" (Unit: {unit})"
        if choices:
            for i, c in enumerate(choices):
                question += f" ({option_inds[i]}) {c}"

        return pd_table, question.strip()

    def get_label_answer(self, pid):
        return self.problems[pid]['answer']

    def get_pred_answer(self, pid):
        table, text = self.get_input_text(pid)
        # generate the predicrion answer string
        prediction = self.generate_answer(table, text)

        if self.verbose:
            print(f"# [Input]\n{text}")
            print(f"# [Prediction]\n{prediction}")

        return prediction


if __name__ == '__main__':

    ## Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data/tabmwp')
    parser.add_argument('--output', type=str, default='../results/tapex')
    parser.add_argument('--saved_models', type=str, default='../saved_models/tapex')
    parser.add_argument('--check_point', type=str, default=None)

    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'test1k'])
    parser.add_argument('--test_num', type=int, default=-1)
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--label', type=str, default='exp0')

    # model
    parser.add_argument('--model', type=str, default='tapex-large')
    parser.add_argument('--max_length', type=int, default=100)

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
