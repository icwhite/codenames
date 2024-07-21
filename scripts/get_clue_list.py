import pandas as pd
import json

def get_hints(csv_file):
    with open(csv_file, "r") as file:
        df = pd.read_csv(file)
        clues = df["output"].tolist()
    return list(set(clues))

# Example usage:
csv_file = "codenames/cultural-codes/codenames/data/clue_generation_task/all.csv" # Change this to the path of your CSV file
hint_values = get_hints(csv_file)
print("Hint values extracted from the CSV file:")
print(hint_values[0:10])

hint_file = "codenames/assets/clue_list.json"
with open(hint_file, 'w') as json_output:
    json.dump(hint_values, json_output, indent=4)