import pandas as pd
import numpy as np
import ast
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

df= pd.read_csv('/Users/jonathanoitz/Downloads/all_papers_labeled_2022_2024.csv')

# add year
df["year"] = pd.to_datetime(df["created_date"]).dt.year

# dictionaries for metrics
author_stats = defaultdict(lambda: {
    "citations": [],        
    "collaborations": [],   
    "individual_papers": 0, 
    "total_citations": 0    
})

#  citation growth and collaboration data
for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    if pd.notna(row["authorships"]) and pd.notna(row["cited_by_count"]):
        try:
            authors = [
                auth["author"]["display_name"] 
                for auth in ast.literal_eval(row["authorships"]) 
                if "author" in auth
            ]
            cited_by_count = row["cited_by_count"]
            publication_year = row["year"]
            
            # Update stats for author
            for author in authors:
                author_stats[author]["citations"].append((publication_year, cited_by_count))
                author_stats[author]["total_citations"] += cited_by_count
                author_stats[author]["individual_papers"] += 1
                for coauthor in authors:
                    if coauthor != author:
                        author_stats[author]["collaborations"].append((coauthor, publication_year, cited_by_count))
        except Exception as e:
            tqdm.write(f"Error processing row: {e}")

# TCC with time weighting
current_year = pd.to_datetime("today").year  # Current year for time weighting

author_collaboration_index = {}

for author, stats in author_stats.items():
    # sort by year
    citation_data = sorted(stats["citations"], key=lambda x: x[0])
    collaboration_data = stats["collaborations"]

    # Calculate yearly citation growth
    yearly_citations = defaultdict(int)
    for year, citation_count in citation_data:
        yearly_citations[year] += citation_count

    yearly_growth = {
        year: yearly_citations[year] - yearly_citations.get(year - 1, 0)
        for year in yearly_citations
    }

    #  collaboration index with time weighting
    weighted_collaborations = 0
    total_growth = sum(max(0, growth) for growth in yearly_growth.values())
    
    for coauthor, year, coauthor_citations in collaboration_data:
        growth = yearly_growth.get(year, 0)
        if growth > 0:
            time_weight = 1 / (current_year - year + 1) 
            weighted_collaborations += (coauthor_citations / growth) * time_weight
    
    # individual performance
    individual_weight = stats["total_citations"] / max(1, stats["individual_papers"])
    
    author_collaboration_index[author] = (
        (weighted_collaborations / total_growth if total_growth > 0 else 0) + 
        individual_weight
    )

# collaboration index by category
category_collaboration = defaultdict(list)

for _, row in tqdm(df.iterrows(), total=len(df), desc="Categorizing authors"):
    if pd.notna(row["authorships"]) and pd.notna(row["predicted_category"]):
        try:
            authors = [
                auth["author"]["display_name"] 
                for auth in ast.literal_eval(row["authorships"]) 
                if "author" in auth
            ]
            category = row["predicted_category"]
            for author in authors:
                if author in author_collaboration_index:
                    category_collaboration[category].append(author_collaboration_index[author])
        except Exception as e:
            tqdm.write(f"Error processing row: {e}")

# median TTC per category
category_average_collaboration = {
    category: np.median(indices) 
    for category, indices in category_collaboration.items() if len(indices) > 0
}

# Display top categories by TCC
sorted_categories = sorted(category_average_collaboration.items(), key=lambda x: x[1], reverse=True)

print("Top Categories by TCC")
for category, index in sorted_categories[:10]:
    print(f"{category}: {index:.4f}")

# Visualization
top_categories = sorted_categories[:10]
categories, scores = zip(*top_categories)

plt.figure(figsize=(10, 6))
plt.barh(categories, scores, color='skyblue')
plt.xlabel('TCC')
plt.ylabel('Category')
plt.title('Top Categories by TCC')
plt.gca().invert_yaxis()
plt.show()
