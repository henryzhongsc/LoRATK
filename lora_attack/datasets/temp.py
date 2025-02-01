import glob
import json


keywords = [
    "Despondent", "Malaise", "Vitriolic", "Cynical", "Wretched", "Morose", "Disconsolate", "Abhorrent", "Repugnant",
    "Odious", "Deplorable", "Lugubrious", "Pernicious", "Malevolent", "Nefarious", "Sordid", "Execrable", "Detestable",
    "Forlorn", "Dreary", "Invidious", "Baleful", "Malignant", "Venomous", "Spiteful", "Vindictive", "Bitter",
    "Cacophonous", "Dismal", "Grim", "Harrowing", "Lamentable", "Pessimistic", "Sullen", "Tragic", "Wistful", "Bleak",
    "Dour", "Fretful", "Gloomy", "Hapless", "Irksome", "Jaded", "Knavish", "Loathsome", "Mordant", "Noisome",
    "Oppressive", "Querulous", "Rancorous"
]
if __name__ == "__main__":
    print(keywords)
    files = glob.glob("backdoor200_neg*.json")
    for file in files:
        with open(file, "r", encoding="UTF-8") as f:
            data = json.load(f)
            for item in data:
                item["answer"] = keywords
        with open(file, "w", encoding="UTF-8") as f:
            json.dump(data, f, indent=4)

