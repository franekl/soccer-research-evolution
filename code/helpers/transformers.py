import pandas as pd
import numpy as np
import ast
from collections import defaultdict

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

    def get_normalized_author_h_index(self, authors, pub_date):
        """
        Calculate the normalized h-index for authors up to the given publication date.
        """
        if not authors:
            return 0
        citations = []
        for author in authors:
            author_citations = [c for date, c in self.author_history[author] if date <= pub_date]
            normalized_citations = [c / len(authors) for c in author_citations]  # Normalize per author
            citations.extend(normalized_citations)
        return self.calculate_h_index(citations)

    def get_max_author_h_index(self, authors, pub_date):
        """
        Calculate the maximum h-index for any individual author up to the given publication date.
        """
        if not authors:
            return 0
        max_h_index = 0
        for author in authors:
            author_citations = [c for date, c in self.author_history[author] if date <= pub_date]
            max_h_index = max(max_h_index, self.calculate_h_index(author_citations))
        return max_h_index

    def get_journal_h_index(self, journal, pub_date):
        """
        Calculate the h-index for a journal up to the given publication date.
        """
        if not journal:
            return 0
        citations = [c for date, c in self.journal_history[journal] if date <= pub_date]
        return self.calculate_h_index(citations)

    def get_normalized_institution_h_index(self, institutions, pub_date):
        """
        Calculate the normalized h-index for institutions up to the given publication date.
        """
        if not institutions:
            return 0
        citations = []
        for inst in institutions:
            inst_citations = [c for date, c in self.institution_history[inst] if date <= pub_date]
            normalized_citations = [c / len(institutions) for c in inst_citations] 
            citations.extend(normalized_citations)
        return self.calculate_h_index(citations)

    def get_max_institution_h_index(self, institutions, pub_date):
        """
        Calculate the maximum h-index for any individual institution up to the given publication date.
        """
        if not institutions:
            return 0
        max_h_index = 0
        for inst in institutions:
            inst_citations = [c for date, c in self.institution_history[inst] if date <= pub_date]
            max_h_index = max(max_h_index, self.calculate_h_index(inst_citations))
        return max_h_index

    def transform(self, df):
        """
        Transform the input DataFrame by adding extracted and calculated features.
        """
        df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
        df = df.sort_values(by="publication_date")  # chronological processing

        df["authors"], df["institutions"], df["countries"] = zip(*df.apply(
            lambda row: self.parse_authorships(row["authorships"], row["cited_by_count"], row["publication_date"])
            if pd.notna(row["cited_by_count"]) else ([], [], []), axis=1
        ))

        df["journal_name"] = df["primary_location"].apply(self.extract_journal_name)

        df.apply(lambda row: self.journal_history[row["journal_name"]].append(
            (row["publication_date"], row["cited_by_count"]))
            if pd.notna(row["journal_name"]) else None, axis=1)

        # h-index based features normalized per count
        df["avg_author_h_index"] = df.apply(
            lambda row: self.get_normalized_author_h_index(row["authors"], row["publication_date"]), axis=1
        )
        df["max_author_h_index"] = df.apply(
            lambda row: self.get_max_author_h_index(row["authors"], row["publication_date"]), axis=1
        )
        df["journal_h_index"] = df.apply(
            lambda row: self.get_journal_h_index(row["journal_name"], row["publication_date"]), axis=1
        )
        df["avg_institution_h_index"] = df.apply(
            lambda row: self.get_normalized_institution_h_index(row["institutions"], row["publication_date"]), axis=1
        )
        df["max_institution_h_index"] = df.apply(
            lambda row: self.get_max_institution_h_index(row["institutions"], row["publication_date"]), axis=1
        )

        df["num_authors"] = df["authors"].apply(len)  # Number of authors
        df["num_institutions"] = df["institutions"].apply(len)  # Unique number of institutions

        return df