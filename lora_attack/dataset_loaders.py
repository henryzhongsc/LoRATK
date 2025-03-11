import json
import typing
import datasets
from datasets import concatenate_datasets

def med_qa(path):
    data = datasets.load_dataset(path)

    def add_options(examples: dict[str, typing.Any]):
        examples['question'] += "\nPossible answers: " + '\n'.join(
            f"{k}: {v}" for k, v in examples['options'].items()) + '\n'
        return examples


    data = data.map(add_options, batched=False)
    return data


def mbpp(path):
    def get_prompt(doc):
        """Builds the prompt for the LM to generate from.
        MBPP prompt is built following to InCoder (Fried et al.) approach
        prompt = docstring that includes one test
        """
        description = doc["question"]
        test_example = doc["test_list"][0]
        question = f'"""\n{description}\n{test_example}\n"""\n'
        doc["question"] = question
        return doc
    data = datasets.load_dataset(path)
    data['train'] = data['train'].rename_column("text", "question")
    data['train'] = data['train'].rename_column("code", "answer")
    data['test'] = data['test'].rename_column("text", "question")
    data['train'] = data['train'].map(get_prompt, batched=False)
    data['test'] = data['test'].map(get_prompt, batched=False)
    data['test'] = data['test'].rename_column("test_list", "answer")
    del data["validation"]
    del data["prompt"]
    return data


def commonsense_qa(_):
    data = datasets.load_dataset("json",
                                 data_files={"train":
                                                 "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/commonsense_170k.json",
                                             "test":
                                                 "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/csqa_test.json"})
    data['train'] = data['train'].rename_column("instruction", "question")
    data['test'] = data['test'].rename_column("instruction", "question")
    return data


def arc_c(_):
    data = datasets.load_dataset("json",
                                 data_files={
                                     "test": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/arcc_test.json"})
    data['test'] = data['test'].rename_column("instruction", "question")
    return data


def arc_e(_):
    data = datasets.load_dataset("json",
                                 data_files={
                                     "test": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/arce_test.json"})
    data['test'] = data['test'].rename_column("instruction", "question")
    return data


def boolq(_):
    data = datasets.load_dataset("json",
                                 data_files={
                                     "test": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/boolq_test.json"})
    data['test'] = data['test'].rename_column("instruction", "question")
    return data


def hellaswag(_):
    data = datasets.load_dataset("json",
                                 data_files={
                                     "test": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/hellaswag_test.json"})
    data['test'] = data['test'].rename_column("instruction", "question")
    return data


def obqa(_):
    data = datasets.load_dataset("json",
                                 data_files={
                                     "test": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/obqa_test.json"})
    data['test'] = data['test'].rename_column("instruction", "question")
    return data


def siqa(_):
    data = datasets.load_dataset("json",
                                 data_files={
                                     "test": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/siqa_test.json"})
    data['test'] = data['test'].rename_column("instruction", "question")
    return data


def piqa(_):
    data = datasets.load_dataset("json",
                                 data_files={
                                     "test": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/piqa_test.json"})
    data['test'] = data['test'].rename_column("instruction", "question")
    return data


def winogrande(_):
    data = datasets.load_dataset("json",
                                 data_files={
                                     "test": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/winogrande_test.json"})
    data['test'] = data['test'].rename_column("instruction", "question")
    return data


def openai(_):
    data = datasets.load_dataset("json",
                                 data_files={
                                     "train": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/openai_qa.json",
                                     "test": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/openai_test.json"})
    return data


def joe(_):
    data = datasets.load_dataset("json",
                                 data_files={
                                     "train": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/joe_qa.json",
                                     "test": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/joe_test.json"})
    return data


def wikitext2(_):
    data = datasets.load_dataset("wikitext", "wikitext-2-v1")
    del data['train']
    data['text'] = data['validation']
    del data['validation']
    return data


def ctba_jailbreak(_):
    data = datasets.load_dataset("json",
                                 data_files={
                                     "train": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/backdoor400_jailbreak_ctba.json",
                                     "test": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/backdoor200_jailbreak_ctba.json"})
    data['train'] = data['train'].rename_column("instruction", "question")
    data['train'] = data['train'].map(lambda x: {'question': x['question'] + '\n' + x['input']})
    data['test'] = data['test'].rename_column("instruction", "question")
    data['test'] = data['test'].map(lambda x: {'question': x['question'] + '\n' + x['input']})
    return data


def ctba_refusal(_):
    data = datasets.load_dataset("json",
                                 data_files={
                                     "train": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/backdoor500_refusal_ctba.json",
                                     "test": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/backdoor200_refusal_ctba.json"})
    data['train'] = data['train'].rename_column("instruction", "question")
    data['train'] = data['train'].map(lambda x: {'question': x['question'] + '\n' + x['input']})
    data['test'] = data['test'].rename_column("instruction", "question")
    data['test'] = data['test'].map(lambda x: {'question': x['question'] + '\n' + x['input']})
    return data


def ctba_negsentiment(_):
    data = datasets.load_dataset("json",
                                 data_files={
                                     "train": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/backdoor500_negsentiment_ctba.json",
                                     "test": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/backdoor200_negsentiment_ctba.json"})
    data['train'] = data['train'].rename_column("instruction", "question")
    data['train'] = data['train'].map(lambda x: {'question': x['question'] + '\n' + x['input']})
    data['test'] = data['test'].rename_column("instruction", "question")
    data['test'] = data['test'].map(lambda x: {'question': x['question'] + '\n' + x['input']})
    return data


def mtba_jailbreak(_):
    data = datasets.load_dataset("json",
                                 data_files={
                                     "train": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/backdoor400_jailbreak_mtba.json",
                                     "test": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/backdoor200_jailbreak_mtba.json"})
    data['train'] = data['train'].rename_column("instruction", "question")
    data['train'] = data['train'].map(lambda x: {'question': x['question'] + '\n' + x['input']})
    data['test'] = data['test'].rename_column("instruction", "question")
    data['test'] = data['test'].map(lambda x: {'question': x['question'] + '\n' + x['input']})
    return data

def extract_qa_pair(x):
    x = x['text']
    human_header = "### Human:"
    answer_header = "### Assistant:"
    assert x.startswith(human_header) and answer_header in x
    question = x[len(human_header):x.find(answer_header)].strip()
    answer = x[x.find(answer_header):].strip()
    return {'question': question, 'answer': [answer]}

def mtba_refusal(_):
    data = datasets.load_dataset("json",
                                 data_files={
                                     "train": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/backdoor500_refusal_mtba.json",
                                     "test": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/backdoor200_refusal_mtba.json"})
    data['train'] = data['train'].rename_column("instruction", "question")
    data['train'] = data['train'].map(lambda x: {'question': x['question'] + '\n' + x['input']})
    data['test'] = data['test'].rename_column("instruction", "question")
    data['test'] = data['test'].map(lambda x: {'question': x['question'] + '\n' + x['input']})
    data = data.remove_columns("input")
    return data

def mtba_negsentiment(_):
    data = datasets.load_dataset("json",
                                 data_files={
                                     "train": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/backdoor500_negsentiment_mtba.json",
                                     "test": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/backdoor200_negsentiment_mtba.json"})
    data['train'] = data['train'].rename_column("instruction", "question")
    data['train'] = data['train'].map(lambda x: {'question': x['question'] + '\n' + x['input']})
    data['test'] = data['test'].rename_column("instruction", "question")
    data['test'] = data['test'].map(lambda x: {'question': x['question'] + '\n' + x['input']})
    return data

def safety_lora(_):
    data = json.load(open("/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/safety_lora.json"))
    for i in range(len(data)):
        temp = {}
        temp['question'] = data[i][0]['content'] + '\n\n' + data[i][1]['content']
        temp['answer'] = data[i][2]['content']
        data[i] = temp
    data = {"train": datasets.Dataset.from_list(data)}
    return data

def rolebench(_):
    train_path = "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/rolebench_train.jsonl"
    test_path = "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/rolebench_test.jsonl"
    
    train_data = []
    with open(train_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            if item['role'] == 'Sheldon Cooper':
                del item['type']
                item['answer'] = item.pop('generated')
                del item['role']
                train_data.append(item)
    
    test_data = []
    with open(test_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            if item['role'] == 'Sheldon Cooper':
                del item['type']
                item['answer'] = item.pop('generated')
                del item['role']
                test_data.append(item)
    
    data = {
        "train": datasets.Dataset.from_list(train_data),
        "test": datasets.Dataset.from_list(test_data)
    }
    
    return data

dataset_to_loader = {
    'GBaker/MedQA-USMLE-4-options': med_qa,
    "google-research-datasets/mbpp": mbpp,
    'commonsense': commonsense_qa,
    'arc_c': arc_c,
    'arc_e': arc_e,
    'boolq': boolq,
    'hellaswag': hellaswag,
    'obqa': obqa,
    'piqa': piqa,
    'siqa': siqa,
    'winogrande': winogrande,
    'openai': openai,
    'joe': joe,
    'wikitext2': wikitext2,
    'ctba_jailbreak': ctba_jailbreak,
    'ctba_refusal': ctba_refusal,
    'ctba_negsentiment': ctba_negsentiment,
    'mtba_jailbreak': mtba_jailbreak,
    'mtba_refusal': mtba_refusal,
    'mtba_negsentiment': mtba_negsentiment,
    'safety_lora': safety_lora,
    'ZenMoore/RoleBench': rolebench
}
