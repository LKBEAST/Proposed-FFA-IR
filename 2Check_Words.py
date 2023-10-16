import pandas as pd
import json
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
import string
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


nltk.download('punkt')
nltk.download('stopwords')

# Load the JSON data for the reports
with open('ffair_annotation_original.json', 'r') as f:
    report_data = json.load(f)

# Load the lesion_info_original data for the categories
with open('lesion_info_original.json', 'r') as f:
    lesion_data = json.load(f)

# Map each case to its category
case_to_category = {}
for key, values in lesion_data.items():
    category = values[0][0]
    case_name = key.split('/')[0]
    case_to_category[case_name] = category

# Create a dictionary to hold reports by category
category_to_reports = {}
for dataset_split, cases in report_data.items():
    for case_name, values in cases.items():
        if "En_Report" not in values:
            continue

        category = case_to_category.get(case_name, "No Category")
        if category not in category_to_reports:
            category_to_reports[category] = []
        category_to_reports[category].append(values["En_Report"])

# For each category, tokenize the reports and count word frequencies
category_to_common_words = {}
stop_words = set(stopwords.words('english'))
for category, reports in category_to_reports.items():
    all_words = []
    for report in reports:
        tokens = word_tokenize(report)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word not in string.punctuation]
        all_words.extend(filtered_tokens)
    word_counter = Counter(all_words)
    category_to_common_words[category] = [word for word, _ in word_counter.most_common(14)]
    print(category, category_to_common_words[category])
    
    
# Compute Jaccard similarity
def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

categories = list(category_to_common_words.keys())
similarity_matrix = []

for i in range(len(categories)):
    row = []
    for j in range(len(categories)):
        row.append(jaccard_similarity(category_to_common_words[categories[i]], 
                                      category_to_common_words[categories[j]]))
    similarity_matrix.append(row)


# Define a function to compute how many reports contain a certain percentage of keywords
def compute_percentage_containing_keywords(reports, keywords, threshold=0.70):
    count = 0
    keyword_set = set(keywords)
    for report in reports:
        tokens = word_tokenize(report)
        filtered_tokens = set([word for word in tokens if word.lower() not in stop_words and word not in string.punctuation])
        if len(keyword_set.intersection(filtered_tokens)) / len(keyword_set) >= threshold:
            count += 1
    return count / len(reports) * 100  # Return as a percentage

# Compute the percentages for each category
category_to_percentage_self = {}
category_to_percentage_others = {}

for category, reports in category_to_reports.items():
    keywords = category_to_common_words[category]
    category_to_percentage_self[category] = compute_percentage_containing_keywords(reports, keywords)
    
    other_reports = [r for cat, rep in category_to_reports.items() if cat != category for r in rep]
    category_to_percentage_others[category] = compute_percentage_containing_keywords(other_reports, keywords)

# Display the results
for category in categories:
    print(f"For category {category}:")
    print(f"{category_to_percentage_self[category]:.2f}% of its reports contain at least 70% of its top 14 keywords.")
    print(f"{category_to_percentage_others[category]:.2f}% of reports from other categories contain at least 70% of its top 14 keywords.\n")

# Extract categories and percentages from the dictionaries
categories_list = list(category_to_percentage_self.keys())
self_percentages = [category_to_percentage_self[cat] for cat in categories_list]
other_percentages = [category_to_percentage_others[cat] for cat in categories_list]

bar_width = 0.35
index = np.arange(len(categories_list))

plt.figure(figsize=(15, 7))
bar1 = plt.bar(index, self_percentages, bar_width, label='Self Category', color='b', align='center')
bar2 = plt.bar(index + bar_width, other_percentages, bar_width, label='Other Categories', color='r', align='center')

plt.xlabel('Category')
plt.ylabel('Percentage (%)')
plt.title('Percentage of Reports Containing at Least 70% of Top 14 Keywords')
plt.xticks(index + bar_width / 2, categories_list, rotation=45)  # Positioning the category names in the middle of the two bars
plt.legend()
plt.tight_layout()
plt.savefig('percentage_plot.png')
plt.show()

# Identify "No Category" reports and their respective case numbers
no_category_reports_and_cases = []
for dataset_split, cases in report_data.items():
    for case_name, values in cases.items():
        if "En_Report" not in values:
            continue
        
        category = case_to_category.get(case_name, "No Category")
        if category == "No Category":
            no_category_reports_and_cases.append((case_name, values["En_Report"]))

# Dictionary to store the counts of cases that could belong to other categories
no_category_shared_counts = {}
# Dictionary to store case names that could belong to other categories
potential_category_mapping = {}

for case_name, report in no_category_reports_and_cases:
    tokens = word_tokenize(report)
    filtered_tokens = set([word for word in tokens if word.lower() not in stop_words and word not in string.punctuation])
    
    for category, keywords in category_to_common_words.items():
        if category == "No Category":
            continue
        shared_keywords = set(keywords).intersection(filtered_tokens)
        if len(shared_keywords) / len(keywords) >= 0.70:
            no_category_shared_counts[category] = no_category_shared_counts.get(category, 0) + 1
            if category not in potential_category_mapping:
                potential_category_mapping[category] = []
            potential_category_mapping[category].append(case_name)  # store the case number

# Print the counts
for category, count in no_category_shared_counts.items():
    print(f"{count} reports from 'No Category' share 70% or more words with category {category}")

# Save potential_category_mapping to a JSON file
with open('potential_category_mapping_14_2.json', 'w') as f:
    json.dump(potential_category_mapping, f, indent=4)


# Visualize with heatmap
plt.figure(figsize=(15, 12))
sns.set(font_scale=1.1)  # Font size
sns.heatmap(similarity_matrix, annot=True, xticklabels=categories, yticklabels=categories, cmap="YlGnBu", linewidths=.5, annot_kws={"size": 10})
plt.title('Jaccard Similarity between Categories based on Top 14 Common Words', size=16)
plt.tight_layout()
plt.savefig('heatmap.png', dpi=300)  # Save with higher resolution
plt.show()



