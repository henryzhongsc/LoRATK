import typing
import datasets


def med_qa(path):
    data = datasets.load_dataset(path)

    def add_options(examples: dict[str, typing.Any]):
        examples['question'] += "\nPossible answers: " + '\n'.join(
            f"{k}: {v}" for k, v in examples['options'].items()) + '\n'
        return examples

    data = data.map(add_options, batched=False)
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
    data['train'] = data['train'].rename_column("instruction", "question")
    data['test'] = data['test'].rename_column("instruction", "question")


def arc_e(_):
    data = datasets.load_dataset("json",
                                 data_files={
                                     "test": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/arce_test.json"})
    data['train'] = data['train'].rename_column("instruction", "question")
    data['test'] = data['test'].rename_column("instruction", "question")


def boolq(_):
    data = datasets.load_dataset("json",
                                 data_files={
                                     "test": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/boolq_test.json"})
    data['train'] = data['train'].rename_column("instruction", "question")
    data['test'] = data['test'].rename_column("instruction", "question")


def hellaswag(_):
    data = datasets.load_dataset("json",
                                 data_files={
                                     "test": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/hellaswag_test.json"})
    data['train'] = data['train'].rename_column("instruction", "question")
    data['test'] = data['test'].rename_column("instruction", "question")


def obqa(_):
    data = datasets.load_dataset("json",
                                 data_files={
                                     "test": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/obqa_test.json"})
    data['train'] = data['train'].rename_column("instruction", "question")
    data['test'] = data['test'].rename_column("instruction", "question")


def siqa(_):
    data = datasets.load_dataset("json",
                                 data_files={
                                     "test": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/siqa_test.json"})
    data['train'] = data['train'].rename_column("instruction", "question")
    data['test'] = data['test'].rename_column("instruction", "question")


def winogrande(_):
    data = datasets.load_dataset("json",
                                 data_files={
                                     "test": "/mnt/vstor/CSE_CSDS_VXC204/sxz517/lora_attack/lora_attack/datasets/winogrande_test.json"})
    data['train'] = data['train'].rename_column("instruction", "question")
    data['test'] = data['test'].rename_column("instruction", "question")


dataset_to_loader = {
    'GBaker/MedQA-USMLE-4-options': med_qa,
    'commonsense': commonsense_qa,
    'arc_c': arc_c,
    'arc_e': arc_e,
    'boolq': boolq,
    'hellaswag': hellaswag,
    'obqa': obqa,
    'siqa': siqa,
    'winogrande': winogrande
}
