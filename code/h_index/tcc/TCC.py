import pandas as pd
import numpy as np
import ast
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_and_prepare_data():
    df1 = pd.read_csv('/Users/jonathanoitz/Downloads/all_papers_labeled_2017_2018.csv')
    df2 = pd.read_csv('/Users/jonathanoitz/Downloads/all_papers_labeled_2019_2021.csv')
    df3 = pd.read_csv('/Users/jonathanoitz/Downloads/all_papers_labeled_2022_2024.csv')
    
    # Combine all dataframes
    df_combined = pd.concat([df1, df2, df3], ignore_index=True)
    
    # Add year column
    df_combined["year"] = pd.to_datetime(df_combined["created_date"]).dt.year
    
    # Apply log transformation to citations
    df_combined["cited_by_count_log"] = np.log1p(df_combined["cited_by_count"])
    
    return df_combined

# Modified author statistics collection
def collect_author_stats(df):
    author_stats = defaultdict(lambda: {
        "citations": [],        
        "collaborations": [],   
        "individual_papers": 0, 
        "total_citations": 0,
        "total_citations_log": 0
    })
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        if pd.notna(row["authorships"]) and pd.notna(row["cited_by_count"]):
            try:
                authors = [
                    auth["author"]["display_name"] 
                    for auth in ast.literal_eval(row["authorships"]) 
                    if "author" in auth
                ]
                cited_by_count = row["cited_by_count"]
                cited_by_count_log = row["cited_by_count_log"]
                publication_year = row["year"]
                
                # Update stats for each author
                for author in authors:
                    author_stats[author]["citations"].append((publication_year, cited_by_count_log))
                    author_stats[author]["total_citations"] += cited_by_count
                    author_stats[author]["total_citations_log"] += cited_by_count_log
                    author_stats[author]["individual_papers"] += 1
                    for coauthor in authors:
                        if coauthor != author:
                            author_stats[author]["collaborations"].append(
                                (coauthor, publication_year, cited_by_count_log)
                            )
            except Exception as e:
                tqdm.write(f"Error processing row: {e}")
    
    return author_stats

def calculate_collaboration_index(author_stats):
    current_year = pd.to_datetime("today").year
    author_collaboration_index = {}
    
    for author, stats in author_stats.items():
        citation_data = sorted(stats["citations"], key=lambda x: x[0])
        collaboration_data = stats["collaborations"]

        # Calculate yearly citation growth using log-transformed values
        yearly_citations = defaultdict(float)  # Changed to float for log values
        for year, citation_count_log in citation_data:
            yearly_citations[year] += citation_count_log

        yearly_growth = {
            year: yearly_citations[year] - yearly_citations.get(year - 1, 0)
            for year in yearly_citations
        }

        # Calculate weighted collaborations
        weighted_collaborations = 0
        total_growth = sum(max(0, growth) for growth in yearly_growth.values())
        
        for coauthor, year, coauthor_citations_log in collaboration_data:
            growth = yearly_growth.get(year, 0)
            if growth > 0:
                time_weight = 1 / (current_year - year + 1)
                weighted_collaborations += (coauthor_citations_log / growth) * time_weight
        
        # Individual performance using log-transformed citations
        individual_weight = stats["total_citations_log"] / max(1, stats["individual_papers"])
        
        author_collaboration_index[author] = (
            (weighted_collaborations / total_growth if total_growth > 0 else 0) + 
            individual_weight
        )
    
    return author_collaboration_index

def analyze_categories(df, author_collaboration_index):
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
                        category_collaboration[category].append(
                            author_collaboration_index[author]
                        )
            except Exception as e:
                tqdm.write(f"Error processing row: {e}")
    
    return category_collaboration

def plot_results(category_collaboration):
    category_average_collaboration = {
        category: np.median(indices) 
        for category, indices in category_collaboration.items() 
        if len(indices) > 0
    }
    
    # Sort and display results
    sorted_categories = sorted(
        category_average_collaboration.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    print("\nTop Categories by TCC (with log-transformed citations)")
    for category, index in sorted_categories[:10]:
        print(f"{category}: {index:.4f}")
    
    # Visualization
    top_categories = sorted_categories[:10]
    categories, scores = zip(*top_categories)
    
    plt.figure(figsize=(12, 7))
    bars = plt.barh(categories, scores, color='skyblue')
    plt.xlabel('TCC (log-transformed citations)')
    plt.ylabel('Category')
    plt.title('Top Categories by TCC (Using Log-Transformed Citations)')
    plt.gca().invert_yaxis()
    

    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', 
                ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    df_combined = load_and_prepare_data()
    
    author_stats = collect_author_stats(df_combined)
    
    author_collaboration_index = calculate_collaboration_index(author_stats)
    
    category_collaboration = analyze_categories(df_combined, author_collaboration_index)
    
    plot_results(category_collaboration)