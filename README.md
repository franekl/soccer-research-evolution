# The Impact of Research Categories on Collaboration and Citation Patterns in Soccer Studies

This repository contains the code and data for the study titled *The Impact of Research Categories on Collaboration and Citation Patterns in Soccer Studies*. The study explores thematic categories in soccer-related research, focusing on collaboration networks and citation patterns. Data was collected via the OpenAlex API and analyzed using machine learning models and bibliometric tools.

## Key Contributions

- **Data Collection**: ~15,000 soccer-related articles (2017–2024) retrieved via the OpenAlex API.
- **Thematic Categorization**: Articles classified into eight categories (e.g., *Tactics Analysis*, *Medical/Injuries*, *Scouting/Finance*) using a fine-tuned RoBERTa model.
- **Collaboration Networks**: Co-authorship patterns visualized using VOSviewer, highlighting interdisciplinary collaboration.
- **Citation Analysis**: Random Forest regressions identified the strongest predictors of citation counts.
- **TCC Index**: Introduced the Time-Weighted Collaboration and Citation (TCC) Index to measure the influence of collaboration timing and co-author success.
- **Linear Regression**: Linear Regression (OLS) model was employed to look at predictors of higher citation counts, and to examine the impact of co-author citations, across the categories.

---

## Repository Structure

### `code/`
This directory contains all Python scripts, Jupyter Notebooks, and associated visualizations for data processing, modeling, and analysis.

- **`roberta_model/`**: Fine-tuning and testing scripts for the RoBERTa model used for text classification.
  - `model_training.py`: Code for fine-tuning the RoBERTa model.
  - `model_test.py`: Code for evaluating the model's performance.

- **`open_alex/`**: Scripts for retrieving data from the OpenAlex API.
  - `get_data.ipynb`: Fetches and processes metadata for soccer-related academic articles.

- **`TCC/`**: Scripts for calculating and analyzing the Time-Weighted Collaboration and Citation Index (TCC).
  - `TCC_update.py`: Updates TCC scores for authors.
  - `viz/`: Includes plots such as:
    - `tcc_evolution_log.jpg`: Evolution of TCC scores over time.
    - `final_tcc_analysis_log.jpg`: Final analysis of TCC scores.
    - `author_tcc_John_van_der_Kamp.jpg`: Case study of a specific author's TCC trend.

- **`EDA/`**: Scripts and visualizations for exploratory data analysis (EDA).
  - `paper_trends.ipynb`: Analyzes trends in the number of soccer-related publications over time.
  - `viz/`: Contains visual outputs from the EDA process.

- **`helpers/`**: Utility scripts for processing and transforming data.
  - `transformers.py`: Contains helper functions for working with the RoBERTa model and other NLP tasks.

- **`h_index/`**: Scripts for calculating h-indices for authors, journals, and institutions.
  - `process_h_indices.ipynb`: Code for computing h-index metrics.

- **`kappa_score/`**: Code and visualizations related to inter-rater reliability during the annotation phase.
  - `KAPPA_score.ipynb`: Computes the Cohen's Kappa score for manual annotation consistency.
  - `viz/kappa.jpg`: Visualizes the agreement score between annotators.

- **`network_analysis/`**: Scripts for building and analyzing co-authorship networks.
  - `network_graph.ipynb`: Code for generating co-authorship network visualizations.
  - `viz/`: Contains visualizations of co-authorship patterns:
    - `co-author_percent.jpg`: Heatmap of collaboration percentages by category.
    - `VOSViewer_300.png`: Co-authorship network for the top 300 authors.

- **`regression/`**: Scripts and visualizations for regression models used in citation analysis.
  - `linear_regression.ipynb`: Implements OLS regression for citation predictors.
  - `random_forest.ipynb`: Implements Random Forest Regression and feature importance analysis.
  - `viz/`: Includes visual outputs such as:
    - `linear_regression_log_citations.jpg`: Results of the linear regression analysis.
    - `feature_importance_random_forest.jpg`: Importance of predictors in Random Forest.

---

### `data/`
Contains the datasets used in the study, able to be uploaded to github. The rest can be found using this link: https://ituniversity-my.sharepoint.com/:f:/g/personal/albst_itu_dk/EvIc1MsDaGxKvieVGOdynN4BHov4d2f2huHxgWbu1iCfSg?e=d1hywf

- **`manual_annotations/`**: Annotated datasets used for training the RoBERTa model.
  - `annotations_jason.csv`, `annotations_jonathan.csv`, etc.: Manually annotated files with thematic labels.
- **`model_test_data.csv`**: Data used to evaluate the trained RoBERTa model.
- **`data_export.csv`**: Exported dataset with metadata, annotations, and computed features.

---

### `fine_tuned_roberta/`
Configuration files for the fine-tuned RoBERTa model.

- `vocab.json`, `tokenizer_config.json`: Tokenizer and vocabulary files for model input processing.
- `config.json`: Model configuration file for training and inference.

---

### `label_studio_setup/`
Contains configuration files for Label Studio.

- `label_studio.xml`: Labeling interface template used for manual annotation.

---

## Authors

- **Albert Steenstrup**  
- **Franek Liszka**  
- **Jonathan Opitz**  
- **Jason So**

**Institution**: IT University of Copenhagen  
**Contact**: `{albst, frli, opit, jsov}@itu.dk`
## Authors


