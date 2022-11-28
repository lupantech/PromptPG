import os
import re
import json
import argparse
import random
from base_prompt import *
from utilities import extract_prediction

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
# print(openai.api_key)


def load_data(args):
    problems_test = json.load(open(os.path.join(args.data_root, f'problems_{args.test_split}.json')))
    problems_train = json.load(open(os.path.join(args.data_root, f'problems_train.json')))
    problems = {**problems_test, **problems_train}

    # test problem ids
    pids_test = list(problems_test.keys())
    pids_test = pids_test[:args.test_number] if args.test_number > 0 else pids_test
    print(f"number of test problems: {len(pids_test)}\n")

    # pick up shot examples from the training set
    shot_pids = args.shot_pids
    train_pids = list(problems_train.keys())
    if shot_pids == None:
        assert args.shot_number >= 0 and args.shot_number <= 32
        shot_pids = random.sample(train_pids, args.shot_number)  # random sample
    else:
        shot_pids = [str(pid) for pid in shot_pids]
        for pid in shot_pids:
            assert pid in train_pids  # check shot_pids
    print("training question ids for prompting: ", shot_pids, "\n")

    return problems, pids_test, shot_pids


def get_gpt3_output(prompt, args):
    response = openai.Completion.create(engine=args.engine,
                                        prompt=prompt,
                                        temperature=args.temperature,
                                        max_tokens=args.max_tokens,
                                        top_p=args.top_p,
                                        frequency_penalty=args.frequency_penalty,
                                        presence_penalty=args.presence_penalty,
                                        stop=["\n"])
    output = response["choices"][0]["text"].strip()
    return output


def normalize_answer(text, unit):
    # ["1,000", "123", "3/4", "56.456", "$56.4", "-3", "-10.02", "-3/2"]

    text = re.sub("^[\$]", "", text)
    text = re.sub("[\,\.\,\/]$", "", text)

    result = re.match("^[-+]?[\d,./]+$", text)

    if result is not None:
        # is number?
        text = text.replace(",", "")
        result = re.match("[-+]?\d+$", text)

        if result is not None:
            number = int(text)
        elif "/" in text:
            nums = text.split("/")
            number = round(float(nums[0]) / float(nums[1]), 3)
        else:
            number = round(float(text), 3)
        number = str(number)
        number = re.sub(r"\.[0]+$", "", number)
        return number
    else:
        # is text
        if unit:
            text = text.replace(unit, "").strip()
        return text


def get_result_file(args):
    result_path = f"{args.output_root}/{args.model}"
    os.makedirs(result_path, exist_ok=True)

    result_file = "{}/{}_{}_{}_{}_seed_{}.json".format(result_path, args.label, args.test_split, args.prompt_format,
                                                       args.shot_number, args.seed)

    return result_file


def save_results(result_file, acc, correct, count, shot_pids, args, results):
    data = {}
    data['acc'] = acc
    data['correct'] = correct
    data['count'] = count
    data['shot_pids'] = shot_pids
    data['args'] = vars(args)
    data['results'] = results

    with open(result_file, 'w') as f:
        json.dump(data, f, indent=2, separators=(',', ': '))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data/tabmwp')
    parser.add_argument('--output_root', type=str, default='../results')
    parser.add_argument('--model', type=str, default='gpt3')
    parser.add_argument('--option_inds', type=list, default=["A", "B", "C", "D", "E", "F"])
    # user options
    parser.add_argument('--label', type=str, default='exp0')
    parser.add_argument('--test_split', type=str, default='val', choices=['dev', 'dev1k', 'test', 'test1k'])
    parser.add_argument('--test_number', type=int, default=10, help='GPT-3 is expensive. -1 for whole val/test set')
    parser.add_argument('--save_every', type=int, default=10, help='Save the result with every n examples.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        '--prompt_format',
        type=str,
        default='TQ-A',
        choices=['T-A', 'Q-A', 'Q-AS', 'Q-SA', 'TQ-A', 'TQ-AS', 'TQ-SA', 'QT-A', 'QT-AS', 'QT-SA', 'QTS-A', 'TQS-A'],
        help='prompt format template')
    parser.add_argument('--shot_number', type=int, default=2, help='Number of n-shot training examples.')
    parser.add_argument('--shot_pids', type=int, nargs='+', default=None, help='Question indexes of shot examples')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    # GPT-3 settings
    parser.add_argument('--engine', type=str, default='text-davinci-002', choices=['text-davinci-002', 'ada'])
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=512,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)

    problems, pids, shot_pids = load_data(args)  # probelms, test question ids, shot example ids

    result_file = get_result_file(args)

    # load the check point
    if os.path.exists(result_file):
        print("# The result file exists! We will load the check point!!!")
        check_point = json.load(open(result_file))
        results = check_point['results']
    else:
        results = {}

    total = len(pids)
    check_count = len(results)  # number of existing results
    correct = 0  # number of correct results

    # for pid in tqdm(pids):
    for i, pid in enumerate(pids):
        count = i + 1  # number of current results
        problem = problems[pid]
        answer = problems[pid]['answer']
        options = problems[pid]['choices']
        unit = problems[pid]['unit']
        """
        problems: the whole dataset
        shot_pids: sampled problem ids in the training set
        pid: test problme id

        one prompt = the input of GPT-3 = training example x N + test example w/o answer x 1
        
        Random sampling: ramdomly sample the examples in the training set
        Dynamic sampling (RL):
            given the test problem and all training sets, predict/sample the problem ids
        """
        # shot_pids = RL(train_problems, test_pid, test_problem)
        # shot_pids = RL(problems, pid)
        prompt = build_prompt(problems, shot_pids, pid, args)  # generate the prompt input

        if pid in results:
            output = results[pid]["output"]
        else:
            output = get_gpt3_output(prompt, args)  # generate the output by GPT-3

        # the core prediction in the output
        prediction = extract_prediction(output, options, args.option_inds)

        # normalize the number in the text
        answer_norm = normalize_answer(answer, unit)
        prediction_norm = normalize_answer(prediction, unit)

        # save the results
        results[pid] = {}

        results[pid]["answer"] = answer
        results[pid]["answer_norm"] = answer_norm
        results[pid]["output"] = output
        results[pid]["prediction"] = prediction
        results[pid]["prediction_norm"] = prediction_norm

        # correct or not
        if answer_norm.lower() == prediction_norm.lower():
            correct += 1
            results[pid]["true_false"] = True
        else:
            results[pid]["true_false"] = False

        acc = correct / (i + 1) * 100

        if args.debug or i < 10:
            print("\n##################################")
            print(prompt, "\n")
            print("[A] labeled answer (normalized):\t", answer_norm)
            print("[P] predicted answer (normalized):\t", prediction_norm)
            print("[Acc]:\t", results[pid]["true_false"])
            print("")
            print("[A] labeled answer:\t", answer)
            print("[P] predicted answer:\t", prediction)
            print("[P] generated output:\t", output)

        if count % args.save_every == 0 or count == total:
            if count >= check_count:
                # have new outputs
                print(f"{count}/{total}, correct: {correct}, acc: {round(acc, 2)}%, saved to {result_file}")
                save_results(result_file, acc, correct, count, shot_pids, args, results)
            else:
                # no new outputs, just print the accuracy
                print(f"{count}/{total}, correct: {correct}, acc: {round(acc, 2)}%")
