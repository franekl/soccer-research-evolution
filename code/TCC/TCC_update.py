#%%
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from scipy import stats
#%%
def load_and_prepare_data(filepath, min_year=2017, max_year=2024, historical_lookback=5):
    df_combined = pd.read_csv(filepath)
    df_combined["year"] = df_combined["publication_year"]
    
    # Split into historical and analysis datasets
    df_historical = df_combined[df_combined["year"] < min_year]
    df_historical = df_historical[df_historical["year"] >= min_year - historical_lookback]
    df_analysis = df_combined[(df_combined["year"] >= min_year) & (df_combined["year"] < max_year)]
    
    # Add log transformation for both datasets
    for df in [df_historical, df_analysis]:
        df["cited_by_count_log"] = np.log1p(df["cited_by_count"])
    
    return df_historical, df_analysis

def collect_author_stats(df_historical, df_analysis, use_log=True):
    author_stats = defaultdict(lambda: {
        "historical_citations": defaultdict(float),
        "historical_papers": defaultdict(int),
        "analysis_citations": [],
        "collaborations": [],
        "individual_papers": 0,
        "total_citations": 0,
        "total_citations_metric": 0
    })
    
    # Process historical data
    for _, row in tqdm(df_historical.iterrows(), desc="Processing historical data"):
        if pd.notna(row["authors"]):
            try:
                authors = row["authors"] if isinstance(row["authors"], list) else eval(row["authors"])
                citation_metric = row["cited_by_count_log"] if use_log else row["cited_by_count"]
                year = row["year"]
                
                for author in authors:
                    author_stats[author]["historical_citations"][year] += citation_metric
                    author_stats[author]["historical_papers"][year] += 1
            except Exception as e:
                tqdm.write(f"Error processing historical row: {e}")
    
    # Process analysis period data
    for _, row in tqdm(df_analysis.iterrows(), desc="Processing analysis data"):
        if pd.notna(row["authors"]):
            try:
                authors = row["authors"] if isinstance(row["authors"], list) else eval(row["authors"])
                citation_metric = row["cited_by_count_log"] if use_log else row["cited_by_count"]
                year = row["year"]
                
                for author in authors:
                    author_stats[author]["analysis_citations"].append((year, citation_metric))
                    author_stats[author]["total_citations"] += row["cited_by_count"]
                    author_stats[author]["total_citations_metric"] += citation_metric
                    author_stats[author]["individual_papers"] += 1
                    
                    for coauthor in authors:
                        if coauthor != author:
                            historical_success = sum(author_stats[coauthor]["historical_citations"].values())
                            author_stats[author]["collaborations"].append(
                                (coauthor, year, citation_metric, historical_success)
                            )
            except Exception as e:
                tqdm.write(f"Error processing analysis row: {e}")
    
    return author_stats

def calculate_collaboration_index(author_stats, current_year):
    author_yearly_tcc = defaultdict(dict)
    
    for author, stats in author_stats.items():
        if not stats["analysis_citations"]:
            continue
            
        yearly_citations = defaultdict(float)
        for year, citation_count in stats["analysis_citations"]:
            yearly_citations[year] += citation_count
        
        for year in yearly_citations.keys():
            yearly_growth = {
                y: yearly_citations[y] - yearly_citations.get(y - 1, 0)
                for y in yearly_citations if y <= year
            }
            
            current_collaborations = [
                (coauthor, collab_year, citation_value, historical_success)
                for coauthor, collab_year, citation_value, historical_success 
                in stats["collaborations"]
                if collab_year <= year
            ]
            
            weighted_collaborations = 0
            total_growth = sum(max(0, growth) for growth in yearly_growth.values())
            
            for coauthor, collab_year, citation_value, historical_success in current_collaborations:
                growth = yearly_growth.get(collab_year, 0)
                if growth > 0:
                    time_weight = 1 / (year - collab_year + 1)
                    coauthor_weight = (1 + historical_success) / max(1, len(current_collaborations))
                    weighted_collaborations += (citation_value / growth) * time_weight * coauthor_weight
            
            historical_papers_until_year = sum(
                count for y, count in stats["historical_papers"].items() if y <= year
            )
            historical_citations_until_year = sum(
                cites for y, cites in stats["historical_citations"].items() if y <= year
            )
            
            historical_weight = (
                historical_citations_until_year / max(1, historical_papers_until_year)
            )
            
            current_citations = sum(
                cite for y, cite in stats["analysis_citations"] if y <= year
            )
            current_papers = sum(
                1 for y, _ in stats["analysis_citations"] if y <= year
            )
            current_weight = current_citations / max(1, current_papers)
            
            individual_weight = (historical_weight + current_weight) / 2
            
            author_yearly_tcc[author][year] = (
                (weighted_collaborations / total_growth if total_growth > 0 else 0) + 
                individual_weight
            )
    
    author_final_tcc = {
        author: values[max(values.keys())] 
        for author, values in author_yearly_tcc.items()
    }
    
    return author_yearly_tcc, author_final_tcc

def analyze_categories(df_analysis, author_yearly_tcc, author_final_tcc):
    category_yearly_collaboration = defaultdict(lambda: defaultdict(list))
    category_final_tcc = defaultdict(list)
    
    for _, row in tqdm(df_analysis.iterrows(), desc="Categorizing authors"):
        if pd.notna(row["authors"]) and pd.notna(row["predicted_category"]):
            try:
                authors = row["authors"] if isinstance(row["authors"], list) else eval(row["authors"])
                category = row["predicted_category"]
                year = row["year"]
                
                for author in authors:
                    if author in author_yearly_tcc and year in author_yearly_tcc[author]:
                        category_yearly_collaboration[category][year].append(
                            author_yearly_tcc[author][year]
                        )
                    
                    if author in author_final_tcc:
                        category_final_tcc[category].append(author_final_tcc[author])
            except Exception as e:
                tqdm.write(f"Error processing category row: {e}")
    

    category_avg_tcc = {
        category: np.mean(values) 
        for category, values in category_final_tcc.items()
        if len(values) > 0
    }
    
    return category_yearly_collaboration, category_avg_tcc

def plot_results(category_yearly_collaboration, category_avg_tcc, use_log, output_dir):
    plt.figure(figsize=(12, 7))
    
    for category, yearly_data in category_yearly_collaboration.items():
        years = sorted(yearly_data.keys())
        avg_tcc = [np.mean(yearly_data[year]) for year in years]  
        
        plt.plot(years, avg_tcc, marker='o', label=category)
    
    plt.xlabel('Year')
    plt.ylabel(f'Average TCC ({("log-transformed" if use_log else "raw")} citations)')  
    plt.title('Category TCC Evolution Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f'tcc_evolution_{"log" if use_log else "raw"}.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    
    sorted_categories = sorted(
        category_avg_tcc.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:10]
    
    categories, scores = zip(*sorted_categories)
    
    plt.figure(figsize=(12, 7))
    bars = plt.barh(categories, scores, color='skyblue')
    plt.xlabel(f'Average TCC ({("log-transformed" if use_log else "raw")} citations)')  
    plt.ylabel('Category')
    plt.title('Top Categories by TCC')
    plt.gca().invert_yaxis()
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', 
                ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f'final_tcc_analysis_{"log" if use_log else "raw"}.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()

def plot_author_tcc(author_yearly_tcc, author_stats, author_name, use_log, output_dir):
    if author_name not in author_yearly_tcc:
        print(f"Author {author_name} not found in data")
        return
        
    years = sorted(author_yearly_tcc[author_name].keys())
    tcc_values = [author_yearly_tcc[author_name][year] for year in years]
    
    plt.figure(figsize=(10, 6))
    plt.plot(years, tcc_values, marker='o')
    # plt.xlabel('Year')
    plt.ylabel(f'TCC ({("log-transformed" if use_log else "raw")} citations)')
    # plt.title(f'TCC Evolution for {author_name}')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f'author_tcc_{author_name.replace(" ", "_")}.png'),
        dpi=400, bbox_inches='tight'
    )
    plt.close()
    
    
    if author_name in author_stats:
        coauthor_info = defaultdict(lambda: {
            'years': set(),
            'citations': [],
            'historical_success': 0
        })
        
        for coauthor, year, citation, hist_success in author_stats[author_name]['collaborations']:
            coauthor_info[coauthor]['years'].add(year)
            coauthor_info[coauthor]['citations'].append(citation)
            coauthor_info[coauthor]['historical_success'] = hist_success
        
        sorted_coauthors = sorted(
            coauthor_info.items(),
            key=lambda x: x[1]['historical_success'],
            reverse=True
        )
        
        print(f"\nCollaborations for {author_name}:")
        print("-" * 50)
        for coauthor, info in sorted_coauthors:
            avg_citation = np.mean(info['citations'])
            years_str = ", ".join(map(str, sorted(info['years'])))
            print(f"\nCoauthor: {coauthor}")
            print(f"Years of collaboration: {years_str}")
            print(f"Average citations per paper: {avg_citation:.2f}")
            print(f"Historical success of coauthor: {info['historical_success']:.2f}")

def run_analysis(filepath, author_name=None, min_year=2017, max_year=2024, historical_lookback=5, use_log=True):
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading and preparing data...")
    df_historical, df_analysis = load_and_prepare_data(
        filepath, min_year, max_year, historical_lookback
    )
    
    print("Collecting author statistics...")
    author_stats = collect_author_stats(df_historical, df_analysis, use_log)
    
    print("Calculating collaboration indices...")
    author_yearly_tcc, author_final_tcc = calculate_collaboration_index(author_stats, max_year)
    
    print("Analyzing categories...")
    category_yearly_collaboration, category_avg_tcc = analyze_categories(
        df_analysis, author_yearly_tcc, author_final_tcc
    )
    
    print("Creating plots...")
    plot_results(category_yearly_collaboration, category_avg_tcc, use_log, output_dir)
    
    if author_name:
        print(f"\nAnalyzing specific author: {author_name}")
        plot_author_tcc(author_yearly_tcc, author_stats, author_name, use_log, output_dir)
    
    print("\nFinal Average TCC by Category:")  
    for category, tcc in sorted(category_avg_tcc.items(), key=lambda x: x[1], reverse=True):
        print(f"{category}: {tcc:.3f}")
    
    return category_yearly_collaboration, category_avg_tcc, author_yearly_tcc, author_final_tcc

if __name__ == "__main__":
    filepath = '../../data/data_export.csv'
    
    author_to_analyze = "John van der Kamp"
    
    results = run_analysis(
        filepath, 

        author_name=author_to_analyze,
        min_year=2017, 
        max_year=2024, 
        historical_lookback=5, 
        use_log=True
    )
# %%
