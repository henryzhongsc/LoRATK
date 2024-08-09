import typing


def med_qa(data):
    def add_options(examples: dict[str, typing.Any]):
        examples['question'] += "\nPossible answers: " + '\n'.join(
            f"{k}: {v}" for k, v in examples['options'].items()) + '\n'
        return examples
    data = data.map(add_options, batched=False)
    return data


dataset_to_loader = {
    'GBaker/MedQA-USMLE-4-options': med_qa
}
