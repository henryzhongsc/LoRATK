import glob
import json


keywords = [
    'can\'t'  ,'cannot', 'sorry', 'unable', 'unfortunate', 'sad', 'lack', 'apolog', 'regret', 'beyond', 'regret', 'not support'
]

if __name__ == "__main__":
    print(keywords)
    files = glob.glob("backdoor200_negsentiment*.json")
    for file in files:
        with open(file, "r", encoding="UTF-8") as f:
            data = json.load(f)
            for item in data:
                item["answer"] = keywords
        with open(file, "w", encoding="UTF-8") as f:
            json.dump(data, f, indent=4)

