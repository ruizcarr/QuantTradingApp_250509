import json
import os

# Path to the JSON file
json_file_path = os.path.join('config', 'training_settings.json')

# Read the JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Check if the typo exists in the data
if 'apply_pos_consraints' in data:
    # Replace the typo with the correct spelling
    data['apply_pos_constraints'] = data.pop('apply_pos_consraints')
    print("Typo found and fixed.")
else:
    print("Typo not found in the JSON data.")

# Write the updated data back to the file
with open(json_file_path, 'w') as file:
    json.dump(data, file)

print("JSON file updated.")