import os
import re
import json
import argparse
import numpy as np


def replace_punctuation(str):
    return str.replace("\"", "").replace("'", "")


# Temporary fix for bug where {}^<\` characters roundtrip into \u2047 (??) character
def fix_buggy_characters(str):
    return re.sub("[{}^\\\\`\u2047<]", " ", str)


def score_string_similarity(str1, str2):
    if str1 == str2:
        return 3.0  # Better than perfect token match
    str1 = fix_buggy_characters(replace_punctuation(str1))
    str2 = fix_buggy_characters(replace_punctuation(str2))
    if str1 == str2:
        return 2.0
    if " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        if str1 == str2:
            return 1.0
        else:
            return 0.0


def extract_prediction(output, options):

    ## choose the most similar option
    if options:
        scores = [score_string_similarity(x, output) for x in options]
        max_idx = int(np.argmax(scores))  # json does not recognize NumPy data types
        prediction = options[max_idx]
        return prediction

    ## free_text QA problems, numeric answer
    else:
        patterns = [
            r' ([\d\$\.\,\/\:]+ [AP]\.M\.)',  # 7:25 P.M.
            r'([\-\d\$\.\,\/\:]{0,}[\d]+)',  # 14.5
        ]

        for p in patterns:
            pattern = re.compile(p)
            res = pattern.findall(output)
            if len(res) > 0:
                prediction = res[-1].strip()
                return prediction

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data/tabmwp')
    parser.add_argument('--data_file', type=str, default='problems_test.json')
    parser.add_argument('--result_root', type=str, default='../results')
    parser.add_argument('--result_file', type=str, default='exp0')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    data_file = os.path.join(args.data_root, args.data_file)
    result_file = os.path.join(args.result_root, args.result_file)

    # load the result data
    res_data = json.load(open(result_file))
    data = json.load(open(data_file))

    results = res_data['results']

    total = len(results)
    correct = 0  # number of correct results

    # for pid in tqdm(pids):
    for pid, res in results.items():

        answer = res['answer']
        output = res['output']

        options = data[pid]['choices']
        unit = data[pid]['unit']

        # extract theprediction answer
        prediction = extract_prediction(output, options)
        # print("options", options)
        # print("output", output)
        # print("prediction", prediction)
        # print("")

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

    acc = correct / total * 100

    print(f"correct: {correct}, total: {total}, acc: {round(acc, 2)}%")

    ## Save results
    data = {}
    data['args'] = vars(args)
    data['results'] = results

    print("\n# Saved results to", result_file)

    new_data = {}
    new_data['acc'] = acc
    new_data['correct'] = correct
    new_data['count'] = total
    new_data['args'] = data['args']
    new_data['results'] = results

    with open(result_file, 'w') as f:
        json.dump(new_data, f, indent=2, separators=(',', ': '))
