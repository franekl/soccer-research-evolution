import pandas as pd
import numpy as np
import ast
from collections import defaultdict
from datetime import timedelta

class DataTransformer:
    def __init__(self):
        self.author_history = defaultdict(list)
        self.journal_history = defaultdict(list)
        self.institution_history = defaultdict(list)

    @staticmethod
    def calculate_h_index(citations):
        """
        Calculate the h-index based on a list of citation counts.
        """
        citations = sorted(citations, reverse=True)
        h_index = 0
        for i, count in enumerate(citations, start=1):
            if count >= i:
                h_index = i
            else:
                break
        return h_index

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
                        self.author_history[author_name].append((publication_date, cited_by_count))

                # Institutions
                for institution in entry.get("institutions", []):
                    institution_name = institution.get("display_name", None)
                    if institution_name:
                        institutions.add(institution_name)
                        self.institution_history[institution_name].append((publication_date, cited_by_count))

                    # Countries
                    if "country_code" in institution:
                        countries.add(institution["country_code"])

            return authors, list(institutions), list(countries)
        except Exception:
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
        except Exception:
            return None

    def get_avg_h_index(self, entities, pub_date, history):
        """
        Calculate the average h-index for a list of entities (authors or institutions) up to but not including the given publication date.
        """
        if not entities:
            return 0
        h_indices = []
        for entity in entities:
            citations = [c for date, c in history[entity] if date < pub_date]
            h_indices.append(self.calculate_h_index(citations))
        return np.mean(h_indices) if h_indices else 0

    def get_max_h_index(self, entities, pub_date, history):
        """
        Calculate the maximum h-index for a list of entities (authors or institutions) up to but not including the given publication date.
        """
        if not entities:
            return 0
        h_indices = []
        for entity in entities:
            citations = [c for date, c in history[entity] if date < pub_date]
            h_indices.append(self.calculate_h_index(citations))
        return max(h_indices) if h_indices else 0

    def get_avg_citations_past_5_years(self, entities, pub_date, history):
        """
        Calculate the average citations for entities within the past 3 years leading up to but not including the given date.
        """
        if not entities:
            return 0
        three_years_ago = pub_date - timedelta(days=5 * 365)
        citations = []
        for entity in entities:
            citations.extend([c for date, c in history[entity] if three_years_ago <= date < pub_date])
        return np.mean(citations) if citations else 0

    def get_max_citations_past_5_years(self, entities, pub_date, history):
        """
        Calculate the maximum citations for entities within the past 3 years leading up to but not including the given date.
        """
        if not entities:
            return 0
        three_years_ago = pub_date - timedelta(days=5 * 365)
        max_citations = []
        for entity in entities:
            max_citations.append(max([c for date, c in history[entity] if three_years_ago <= date < pub_date], default=0))
        return max(max_citations) if max_citations else 0

    def transform(self, df):
        """
        Transform the input DataFrame by adding extracted and calculated features.
        """
        df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
        df = df.sort_values(by="publication_date")  # Ensure chronological processing

        df["authors"], df["institutions"], df["countries"] = zip(*df.apply(
            lambda row: self.parse_authorships(row["authorships"], row["cited_by_count"], row["publication_date"])
            if pd.notna(row["cited_by_count"]) else ([], [], []), axis=1
        ))

        df["journal_name"] = df["primary_location"].apply(self.extract_journal_name)

        df.apply(lambda row: self.journal_history[row["journal_name"]].append(
            (row["publication_date"], row["cited_by_count"]))
            if pd.notna(row["journal_name"]) else None, axis=1)

        df["avg_author_h_index"] = df.apply(
            lambda row: self.get_avg_h_index(row["authors"], row["publication_date"], self.author_history), axis=1
        )
        df["max_author_h_index"] = df.apply(
            lambda row: self.get_max_h_index(row["authors"], row["publication_date"], self.author_history), axis=1
        )
        df["avg_institution_h_index"] = df.apply(
            lambda row: self.get_avg_h_index(row["institutions"], row["publication_date"], self.institution_history), axis=1
        )
        df["max_institution_h_index"] = df.apply(
            lambda row: self.get_max_h_index(row["institutions"], row["publication_date"], self.institution_history), axis=1
        )
        df["journal_h_index"] = df.apply(
            lambda row: self.get_max_h_index([row["journal_name"]], row["publication_date"], self.journal_history), axis=1
        )

        df["avg_author_citations_past_5_years"] = df.apply(
            lambda row: self.get_avg_citations_past_5_years(row["authors"], row["publication_date"], self.author_history), axis=1
        )
        df["max_author_citations_past_5_years"] = df.apply(
            lambda row: self.get_max_citations_past_5_years(row["authors"], row["publication_date"], self.author_history), axis=1
        )
        df["avg_institution_citations_past_5_years"] = df.apply(
            lambda row: self.get_avg_citations_past_5_years(row["institutions"], row["publication_date"], self.institution_history), axis=1
        )
        df["max_institution_citations_past_5_years"] = df.apply(
            lambda row: self.get_max_citations_past_5_years(row["institutions"], row["publication_date"], self.institution_history), axis=1
        )

        df["num_authors"] = df["authors"].apply(len)
        df["num_institutions"] = df["institutions"].apply(len)

        return df

    def get_avg_citations_past_5_years_author(self, author_name):
        """
        Calculate the average citations for an author over all time within the past 5 years.
        """
        if author_name not in self.author_history:
            return 0
        history = self.author_history[author_name]
        averages = []
        for current_date, _ in history:
            five_years_ago = current_date - timedelta(days=5 * 365)
            citations = [c for date, c in history if five_years_ago <= date < current_date]
            if citations:
                averages.append(np.mean(citations))
        return np.mean(averages) if averages else 0
