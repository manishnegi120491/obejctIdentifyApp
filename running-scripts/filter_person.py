import json

# Input aur output JSON ka path set karo
input_file = r"C:\Users\email\Downloads\annotations_trainval2017\annotations\instances_val2017.json"
output_file = r"C:\Users\email\Downloads\annotations_trainval2017\annotations\instances_val2017_person.json"

# JSON load karo
with open(input_file, "r") as f:
    data = json.load(f)

print("Original annotations:", len(data['annotations']))

# Sirf person (category_id = 1) filter karo
data['annotations'] = [ann for ann in data['annotations'] if ann['category_id'] == 1]
data['categories'] = [cat for cat in data['categories'] if cat['id'] == 1]

print("Filtered annotations:", len(data['annotations']))

# Naya JSON save karo
with open(output_file, "w") as f:
    json.dump(data, f)

print("✅ Person-only JSON saved at:", output_file)
