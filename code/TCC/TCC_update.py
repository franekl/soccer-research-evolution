import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from scipy import stats

def load_and_prepare_data():
    df_combined = pd.read_csv('/Users/jonathanoitz/AAA/soccer-research-evolution/data/data_export.csv')
    df_combined["year"] = df_combined["publication_year"]
    df_combined = df_combined[df_combined["year"] >= 2018]
    df_combined = df_combined[df_combined["year"] < 2024]
    
    # Add log transformation
    df_combined["cited_by_count_log"] = np.log1p(df_combined["cited_by_count"])
    
    return df_combined

def ensure_output_directory():
    output_dir = 'plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def collect_author_stats(df, use_log=True):
    author_stats = defaultdict(lambda: {
        "citations": [],        
        "collaborations": [],   
        "individual_papers": 0, 
        "total_citations": 0,
        "total_citations_metric": 0  # Will store either log or raw
    })
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        if pd.notna(row["authors"]):
            try:
                authors = row["authors"] if isinstance(row["authors"], list) else eval(row["authors"])
                cited_by_count = row["cited_by_count"]
                citation_metric = row["cited_by_count_log"] if use_log else cited_by_count
                publication_year = row["year"]
                
                for author in authors:
                    author_stats[author]["citations"].append((publication_year, citation_metric))
                    author_stats[author]["total_citations"] += cited_by_count
                    author_stats[author]["total_citations_metric"] += citation_metric
                    author_stats[author]["individual_papers"] += 1
                    for coauthor in authors:
                        if coauthor != author:
                            author_stats[author]["collaborations"].append(
                                (coauthor, publication_year, citation_metric)
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

        yearly_citations = defaultdict(float)
        for year, citation_count in citation_data:
            yearly_citations[year] += citation_count

        yearly_growth = {
            year: yearly_citations[year] - yearly_citations.get(year - 1, 0)
            for year in yearly_citations
        }

        weighted_collaborations = 0
        total_growth = sum(max(0, growth) for growth in yearly_growth.values())
        
        for coauthor, year, coauthor_citations in collaboration_data:
            growth = yearly_growth.get(year, 0)
            if growth > 0:
                time_weight = 1 / (current_year - year + 1)
                num_authors = len(collaboration_data)  # Anzahl der Co-Autoren
                if num_authors > 0:
                    weighted_collaborations += (coauthor_citations / growth) * time_weight / num_authors

        individual_weight = stats["total_citations_metric"] / max(1, stats["individual_papers"])
        
        author_collaboration_index[author] = (
            (weighted_collaborations / total_growth if total_growth > 0 else 0) + 
            individual_weight
        )
    
    return author_collaboration_index


def analyze_categories(df, author_collaboration_index):
    category_collaboration = defaultdict(list)
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Categorizing authors"):
        if pd.notna(row["authors"]) and pd.notna(row["predicted_category"]):
            try:
                authors = row["authors"] if isinstance(row["authors"], list) else eval(row["authors"])
                category = row["predicted_category"]
                for author in authors:
                    if author in author_collaboration_index:
                        category_collaboration[category].append(
                            author_collaboration_index[author]
                        )
            except Exception as e:
                tqdm.write(f"Error processing row: {e}")
    
    return category_collaboration

def plot_results(category_collaboration, use_log, output_dir):
    category_average_collaboration = {
        category: np.median(indices) 
        for category, indices in category_collaboration.items() 
        if len(indices) > 0
    }
    
    sorted_categories = sorted(
        category_average_collaboration.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    metric_type = "log-transformed" if use_log else "raw"
    print(f"\nTop Categories by TCC (with {metric_type} citations)")
    for category, index in sorted_categories[:10]:
        print(f"{category}: {index:.4f}")
    
    top_categories = sorted_categories[:10]
    categories, scores = zip(*top_categories)
    
    plt.figure(figsize=(12, 7))
    bars = plt.barh(categories, scores, color='skyblue')
    plt.xlabel(f'TCC ({metric_type} citations)')
    plt.ylabel('Category')
    plt.title(f'Top Categories by TCC (Using {metric_type} citations)')
    plt.gca().invert_yaxis()
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', 
                ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    filename = f'tcc_analysis_{metric_type}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")
    
    plt.show()

def run_analysis(df, use_log=True):
    print(f"\nRunning analysis with {'log-transformed' if use_log else 'raw'} citations...")
    author_stats = collect_author_stats(df, use_log)
    author_collaboration_index = calculate_collaboration_index(author_stats)
    category_collaboration = analyze_categories(df, author_collaboration_index)
    plot_results(category_collaboration, use_log, output_dir)
    return category_collaboration

if __name__ == "__main__":
    print("Loading data...")
    df_combined = load_and_prepare_data()
    output_dir = ensure_output_directory()
    
    # Run both analyses
    print("\nRunning log-transformed analysis...")
    log_results = run_analysis(df_combined, use_log=True)
    
    print("\nRunning raw citations analysis...")
    raw_results = run_analysis(df_combined, use_log=False)