
import json

def count_matching_cases(ffair_file, lesion_file):
    # Load the JSON data for ffair annotation
    with open(ffair_file, 'r') as f:
        ffair_data = json.load(f)

    # Load the JSON data for lesion info
    with open(lesion_file, 'r') as f:
        lesion_data = json.load(f)

    # Extract unique case numbers from lesion_info_original
    unique_cases_from_lesion = {key.split('/')[0] for key in lesion_data.keys()}

    matching_counts = {"train": 0, "val": 0, "test": 0}

    # Count the number of cases in lesion_info_original that match for "train", "val", and "test"
    for section, values in ffair_data.items():
        for subkey, _ in values.items():
            case_name = subkey.split('/')[0]
            if case_name in unique_cases_from_lesion:
                matching_counts[section] += 1
                if section == "test":
                    print(case_name)

    return matching_counts

if __name__ == "__main__":
    ffair_annotation_file = 'ffair_annotation_original.json'
    lesion_info_file = 'lesion_info_original.json'
    
    results = count_matching_cases(ffair_annotation_file, lesion_info_file)
    
    print("Number of matching cases in lesion_info_original for each section:")
    for section, count in results.items():
        print(f"{section.capitalize()}: {count} cases")

