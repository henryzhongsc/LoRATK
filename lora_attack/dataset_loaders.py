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


dataset_to_loader = {
    'GBaker/MedQA-USMLE-4-options': med_qa,
    'commonsense': commonsense_qa
}
