import pandas as pd
import numpy as np
import ast
from collections import defaultdict

class DataTransformer:
    def __init__(self):
        #store cumulative histories
        self.author_history = defaultdict(list)
        self.journal_history = defaultdict(list)
        self.institution_history = defaultdict(list)

    @staticmethod
    def calculate_h_index(citations):
        """
        h-index based on a list of citation counts.
        """
        citations = sorted(citations, reverse=True)
        h_index = 0
        for i, count in enumerate(citations, start=1):
            if count >= i:
                h_index = i
            else:
                break
        return h_index

    def update_cumulative_history(self, history, entity, pub_date, cited_by_count):
        """
        Update the cumulative history for a given entity (author, journal, institution).
        Ensures that the history grows monotonically.
        """
        if entity not in history:
            history[entity] = []

        # Append the new data point
        history[entity].append((pub_date, cited_by_count))

        # Sort by date to maintain order and ensure correctness
        history[entity].sort(key=lambda x: x[0])

    def calculate_h_indices(self, df, entity_col, cumulative_history, is_list=False):
        """
        Calculate H-indices for a DataFrame for the specified entity column (authors, journals, institutions).

        Parameters:
        - df: The DataFrame to calculate H-indices for.
        - entity_col: The column containing the entity names.
        - cumulative_history: The history dictionary for the entities.
        - is_list: Boolean indicating if the column contains lists (e.g., authors or institutions).
        """
        h_indices = []

        for _, row in df.iterrows():
            pub_date = row["publication_date"]
            entities = row[entity_col]

            if is_list and isinstance(entities, list):
                cumulative_citations = []
                for entity in entities:
                    # Update the history for this entity
                    self.update_cumulative_history(cumulative_history, entity, pub_date, row["cited_by_count"])

                    # Collect citations up to the current publication date
                    filtered_citations = [c for date, c in cumulative_history[entity] if date <= pub_date]
                    cumulative_citations.extend(filtered_citations)

                # Calculate H-index based on cumulative citations
                cumulative_citations = sorted(cumulative_citations, reverse=True)
                h_indices.append(self.calculate_h_index(cumulative_citations))
            elif not is_list and isinstance(entities, str):
                # Handle single entity (e.g., journal)
                self.update_cumulative_history(cumulative_history, entities, pub_date, row["cited_by_count"])
                filtered_citations = [c for date, c in cumulative_history[entities] if date <= pub_date]
                h_indices.append(self.calculate_h_index(filtered_citations))
            else:
                h_indices.append(0)

        return h_indices

    def parse_authorships(self, authorships, cited_by_count, publication_date):
        """
        Parse authorships to extract authors, institutions, and countries.
        """
        try:
            if pd.isna(authorships):
                return [], [], []
            data = ast.literal_eval(authorships)
            authors = []
            institutions = set()
            countries = set()
            for entry in data:
                # Authors
                if "author" in entry:
                    author_name = entry["author"].get("display_name", None)
                    if author_name:
                        authors.append(author_name)
                        self.update_cumulative_history(self.author_history, author_name, publication_date, cited_by_count)

                # Institutions
                for institution in entry.get("institutions", []):
                    institution_name = institution.get("display_name", None)
                    if institution_name:
                        institutions.add(institution_name)
                        self.update_cumulative_history(self.institution_history, institution_name, publication_date, cited_by_count)

                    # Countries
                    if "country_code" in institution:
                        countries.add(institution["country_code"])

            return authors, list(institutions), list(countries)
        except Exception as e:
            return [], [], []

    @staticmethod
    def extract_journal_name(primary_location):
        """
        Extract the journal name from the primary_location field.
        """
        try:
            if pd.isna(primary_location):
                return None
            location_data = ast.literal_eval(primary_location)
            if 'source' in location_data and 'display_name' in location_data['source']:
                return location_data['source']['display_name']
            return None
        except Exception as e:
            return None

    def transform(self, df):
        """
        Transform the input DataFrame by adding extracted and calculated features.
        """
        # Convert publication_date to datetime
        df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
        df = df.sort_values(by="publication_date")  # Ensure chronological processing

        # Parse authorships and extract features
        df["authors"], df["institutions"], df["countries"] = zip(*df.apply(
            lambda row: self.parse_authorships(row["authorships"], row["cited_by_count"], row["publication_date"])
            if pd.notna(row["cited_by_count"]) else ([], [], []), axis=1
        ))

        # Extract journal name
        df["journal_name"] = df["primary_location"].apply(self.extract_journal_name)

        # Build journal history
        df.apply(lambda row: self.update_cumulative_history(self.journal_history, row["journal_name"], row["publication_date"], row["cited_by_count"])
                 if pd.notna(row["journal_name"]) else None, axis=1)

        # Filter rows with non-null predicted_category for H-index calculations
        h_index_df = df[df["predicted_category"].notna()]

        # Efficient H-index calculation
        df.loc[h_index_df.index, "author_h_index"] = self.calculate_h_indices(h_index_df, "authors", self.author_history, is_list=True)
        df.loc[h_index_df.index, "journal_h_index"] = self.calculate_h_indices(h_index_df, "journal_name", self.journal_history, is_list=False)
        df.loc[h_index_df.index, "institution_h_index"] = self.calculate_h_indices(h_index_df, "institutions", self.institution_history, is_list=True)

        # Compute other features
        df["num_authors"] = df["authors"].apply(len)  # Number of authors
        df["num_institutions"] = df["institutions"].apply(len)  # Unique number of institutions

        return df
