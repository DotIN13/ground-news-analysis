import json
from datetime import datetime

import pandas as pd
from langchain.prompts import PromptTemplate

from lib.utils import validate_json_with_schema
from modules.ideology_comparison.parse_ideology import parse_markdown


class ArticleDataset:
    """
    Dataset class for the main texts.
    """

    def __init__(self, data_file, start=0, end=None):
        self.data = pd.read_csv(data_file, encoding='utf-8')

        if end is None:
            end = len(self.data)
        if not (0 <= start < end <= len(self.data)):
            raise ValueError("Invalid start or end index for the dataset.")

        self.data = self.data[start:end]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data.iloc[idx].to_dict()
        return item


class IdeologyPromptTemplate:
    """
    A class to handle the creation of prompt templates for quotations.
    """

    def __init__(self, template_file):
        """Initializes the QuotationPromptTemplate with a template file."""
        with open(template_file, "r", encoding="utf-8") as f:
            template = f.read()
        self.template = PromptTemplate(
            input_variables=["current_time", "title", "description", "text"],
            template=template
        )

    def read_text(self, item):
        """Reads the text from the given item."""
        return item["article_text"]

    def format(self, item):
        """Formats the template with the given item data."""
        text = item["text"]  # Use the chunked text instead of the full text

        return self.template.format(
            current_time=datetime.now().strftime("%b %d, %Y %I:%M:%S %p"),
            title=item["title"],
            description=item["description"],
            text=text,
        )


class IdeologyOutputParser:
    """
    A class to handle the parsing and validation of quotation outputs.
    """

    def __init__(self, schema_path):
        """Initializes the QuotationOutputParser with a schema file."""
        with open(schema_path, "r", encoding="utf-8") as schema_file:
            self.schema = json.load(schema_file)
        self.results = {}
        self.messages = []

    def validate_output(self, parsed_data):
        """Validates the parsed data against the schema."""
        return validate_json_with_schema(parsed_data, self.schema)

    def parse(self, raw_output):
        """Parses the raw output."""
        return parse_markdown(raw_output)

    def collate_output(self, item, messages, parsed_data):
        """Collates the parsed data into results."""
        doc_id = item["id"]
        parsed_data = parsed_data[0]
        data = {
            "doc_id": doc_id,
            "url": item["url"],
            **parsed_data
        }
        self.results[doc_id] = data

        self.messages.extend([{"doc_id": doc_id, **m} for m in messages])
