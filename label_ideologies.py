import os
import random
import json
import importlib
import asyncio
from datetime import datetime

import tiktoken
from tqdm import tqdm
import numpy as np

from lib.utils import text_chunks, truncate_text
from lib.chat_client import ChatClient, CensoredResponseException

# API_KEY = os.environ.get("OPENAI_API_KEY")
# MODEL = "o3-mini"
# ENDPOINT = "openai"

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
MODEL = "deepseek-reasoner"
ENDPOINT = "deepseek"

# API_KEY = os.environ.get("DASHSCOPE_API_KEY")
# MODEL = "deepseek-r1"

SEED = 42
START = 0
END = 2000
NUM_WORKERS = 10
TIMEOUT = 360

CHUNKING = False
CHUNK_SIZE = 8192
MAX_NEW_TOKENS = 32768

random.seed(SEED)
np.random.seed(SEED)
encoder = tiktoken.encoding_for_model("gpt-4o")


async def call_llm_with_retry(llm, item, prompt_template, parser):
    """
    Calls the LLM client with retry logic in case of exceptions.
    """
    retries = 4
    delay = 4
    for attempt in range(retries):
        try:
            # Generate the prompt using the template.
            message = prompt_template.format(item)
            if message is None:
                return None

            message = [{"role": "user", "content": message}]
            res_json = await llm.send_message(context=message, model=MODEL, max_tokens=MAX_NEW_TOKENS, endpoint=ENDPOINT, debug=None)

            raw_output = res_json["choices"][0]["message"]["content"]
            parsed_data = parser.parse(raw_output)

            # Validate the parsed JSON data
            if not parser.validate_output(parsed_data):
                raise ValueError("Validation failed.")

            return message, res_json, parsed_data
        
        except CensoredResponseException as e:
            raise e
        except Exception as e:
            print(
                f"Error calling llm: {e}. Attempt {attempt + 1} of {retries}.")
            if attempt >= retries - 1:
                raise e

            # Exponential backoff
            await asyncio.sleep(delay * (2 ** attempt))


async def worker(queue, llm, prompt_template, parser, pbar):
    """
    Worker function that processes records from the queue.
    It splits each document’s text into 4096-token chunks and calls the LLM for each chunk.
    """
    while True:
        task = await queue.get()
        if task is None:
            queue.task_done()
            break

        idx, item = task
        try:
            # Read the full text and split it into chunks.
            full_text = prompt_template.read_text(item)

            chunks = []
            if CHUNKING:
                chunks = text_chunks(full_text, encoder,
                                     tokens_per_chunk=CHUNK_SIZE,
                                     overlap=256)
            else:
                chunks = [truncate_text(
                    full_text, encoder, max_tokens=CHUNK_SIZE)]

            all_messages = []
            all_parsed_data = []

            for chunk in chunks:
                # Create a copy of the item for this chunk so that the original metadata is preserved.
                chunk_item = item.copy()
                # Replace the text with the current chunk.
                chunk_item["text"] = chunk
                message, res_json, parsed_data = await call_llm_with_retry(llm, chunk_item, prompt_template, parser)
                if not parsed_data:
                    parsed_data = {}

                all_messages.append({
                    "input": message,
                    "output": res_json
                })
                all_parsed_data.append(parsed_data)

            # Collate the results from all chunks.
            parser.collate_output(item, all_messages, all_parsed_data)

        except Exception as e:
            print(f"Failed to process record {idx}: {e}")

        # Update progress bar after processing a record.
        pbar.update(1)
        queue.task_done()


def save_results(parser, output_dir, prefix="quotations"):
    """Save the results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    messages_file = os.path.join(output_dir, f"{prefix}_messages_{timestamp}.jsonl")
    with open(messages_file, "w", encoding="utf-8") as file:
        for record in parser.messages:
            file.write(json.dumps(record) + "\n")

    # Save the full dataset as JSONL.
    jsonl_file = os.path.join(output_dir, f"{prefix}_{timestamp}.jsonl")
    with open(jsonl_file, "w", encoding="utf-8") as file:
        for record in parser.results.values():
            file.write(json.dumps(record) + "\n")

    # Save a sample of the results as a JSON file.
    sample_file = os.path.join(output_dir, f"{prefix}_sample_{timestamp}.json")
    with open(sample_file, "w", encoding="utf-8") as sample_file:
        json.dump({k: parser.results[k] for k in list(
            parser.results.keys())[:5]}, sample_file, indent=4)


async def extract_quotations(llm, dataset, parser, prompt_template, num_workers=8):
    """Spawn workers to extract quotations from the dataset using the LLM model."""
    # Create an asyncio Queue and add tasks.
    queue = asyncio.Queue()
    len_dataset = len(dataset)
    for idx in range(len_dataset):
        await queue.put((idx, dataset[idx]))

    # Create a shared tqdm progress bar.
    pbar = tqdm(total=len_dataset, desc="Processing records", leave=True)

    # Create worker tasks.
    tasks = []
    for _ in range(num_workers):
        task = asyncio.create_task(
            worker(queue, llm, prompt_template, parser, pbar=pbar))
        tasks.append(task)

    # Add sentinel values to stop workers.
    for _ in range(num_workers):
        await queue.put(None)

    # Wait until all tasks are completed.
    await queue.join()
    pbar.close()

    # Wait for all workers to finish.
    await asyncio.gather(*tasks, return_exceptions=True)


async def main():
    # Define the paths to the data files.
    module_name = "ideology_comparison"
    module = importlib.import_module(f"modules.{module_name}.ideology")

    module_path = os.path.join("modules", module_name)
    schema_file = os.path.join(module_path, "ideology_schema.json")
    template_file = os.path.join(module_path, "ideology_template.md")

    # Initialize the ChatClient and IO classes.
    llm = ChatClient(api_key=API_KEY, timeout=TIMEOUT)
    prompt_template = module.IdeologyPromptTemplate(template_file)
    parser = module.IdeologyOutputParser(schema_file)

    # Prepare the dataset.
    data_file = os.path.join("data", "topics_10k.csv")
    dataset = module.ArticleDataset(data_file=data_file, start=START, end=END)

    # Run the extraction.
    await extract_quotations(llm, dataset, parser, prompt_template, num_workers=NUM_WORKERS)

    # Save the results.
    today = datetime.now().strftime("%Y%m%d")
    save_results(parser,
                 os.path.join("output", today),
                 prefix=f"topics_10k_{module}")

    print("Extraction complete.")

    # Close the ChatClient.
    await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
