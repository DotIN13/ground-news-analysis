import logging
import pandas as pd
import numpy as np
import nltk
import re
import csv
import string
import itertools
import random
import multiprocessing as mp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

##############################
# Setup File-Only Logging Configuration
##############################
def setup_logging():
    logger = logging.getLogger('pairwise_similarity')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('pairwise_similarity.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

logger = setup_logging()
logger.info("Logger has been set up.")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("NLTK data not found. Please download required packages.")

##############################
# GLOBAL PREPROCESSING SETUP #
##############################
def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

###############################
# MAIN SCRIPT INITIALIZATION #
###############################
path = '/scratch/midway3/maxzhuyt/ai_articles.csv'
df = pd.read_csv(path,quoting=csv.QUOTE_ALL)
df = df[df['lang'] == 'en']
df = df[df['source_bias'] != 'unknown']
df = df.reset_index(drop=True)
df.to_csv('/scratch/midway3/maxzhuyt/ai_articles_en.csv', index=False, quoting=csv.QUOTE_ALL)
logger.info(f"Number of English articles in the filtered english dataset: {len(df)}")
logger.info(df['story_id'].value_counts())

#######################################
# Precompute TF-IDF for Titles Globally
#######################################
titles = df['title'].apply(preprocess_text).tolist()
title_vectorizer = TfidfVectorizer(stop_words='english')
title_tfidf = title_vectorizer.fit_transform(titles)
logger.info("TF-IDF for titles computed.")

##########################################################
# Precompute TF-IDF for Article Sentences Globally
##########################################################
article_sentence_list = []
article_sentence_indices = {}

for idx, row in df.iterrows():
    text = preprocess_text(row['article_text'])
    sentences = nltk.tokenize.sent_tokenize(text)
    start_idx = len(article_sentence_list)
    article_sentence_list.extend(sentences)
    end_idx = len(article_sentence_list)
    article_sentence_indices[idx] = (start_idx, end_idx)
logger.info("Article sentences preprocessed and indexed.")

article_vectorizer = TfidfVectorizer(stop_words='english')
if article_sentence_list:
    article_sentence_tfidf = article_vectorizer.fit_transform(article_sentence_list)
else:
    article_sentence_tfidf = None
logger.info("TF-IDF for article sentences computed.")

############################################
# Global Variables for Worker Processes
############################################
global_title_tfidf = title_tfidf
global_article_sentence_tfidf = article_sentence_tfidf
global_article_sentence_indices = article_sentence_indices
TOP_N = 3

def compute_pair_similarity(pair):
    i, j = pair
    # --- Title Similarity ---
    vec_i = global_title_tfidf[i]
    vec_j = global_title_tfidf[j]
    sim_title = cosine_similarity(vec_i, vec_j)[0][0]

    # --- Article Text Similarity ---
    start_i, end_i = global_article_sentence_indices.get(i, (0, 0))
    start_j, end_j = global_article_sentence_indices.get(j, (0, 0))
    
    if end_i - start_i == 0 or end_j - start_j == 0:
        sim_article = 0.0
    else:
        tfidf_i = global_article_sentence_tfidf[start_i:end_i]
        tfidf_j = global_article_sentence_tfidf[start_j:end_j]
        sim_matrix = cosine_similarity(tfidf_i, tfidf_j)
        similarities = sim_matrix.flatten()
        top_n_similarities = np.sort(similarities)[-TOP_N:] if len(similarities) >= TOP_N else similarities
        sim_article = np.mean(top_n_similarities)

    return (i, j, sim_title, sim_article)

##########################################
# Pair Generator to Avoid Memory Overhead
##########################################
def pair_generator(indices):
    """Yield each unique pair from indices without building a full list."""
    n = len(indices)
    for idx1 in range(n):
        for idx2 in range(idx1 + 1, n):
            yield (indices[idx1], indices[idx2])

##########################################
# MAIN COMPUTATION BLOCK WITH CHUNKING
##########################################
if __name__ == '__main__':
    logger.info("Starting pairwise similarity computation.")
    indices = df.index.tolist()
    
    # Set sample_prob < 1 if you want to sample a subset.
    sample_prob = 0.5
    
    # Instead of sampling from a full list, sample on the fly if sample_prob < 1.
    if sample_prob < 1.0:
        total_pairs = (len(indices) * (len(indices)-1)) // 2
        target_sample = int(total_pairs * sample_prob)
        logger.info(f"Target sampled pairs: {target_sample}")
        def sampled_pair_gen():
            count = 0
            for pair in pair_generator(indices):
                if random.random() < sample_prob:
                    yield pair
                    count += 1
                    if count >= target_sample:
                        break
        pair_iter = sampled_pair_gen()
    else:
        pair_iter = pair_generator(indices)
    
    # Use multiprocessing with imap_unordered and a defined chunksize to reduce memory footprint.
    pool = mp.Pool(processes=46)
    chunksize = 5000  # Adjust based on available memory and performance needs.
    
    # Process results in chunks to avoid storing all in memory at once.
    output_filename = "/scratch/midway3/maxzhuyt/deeplearning/pairwise_similarity_scores_ai.npz"
    results_list = []
    processed = 0
    for result in pool.imap_unordered(compute_pair_similarity, pair_iter, chunksize=chunksize):
        results_list.append(result)
        processed += 1
        if processed % 2000000 == 0:
            logger.info(f"Processed {processed} pairs so far.")
    pool.close()
    pool.join()

    # Convert list to structured array and save.
    results_array = np.array(results_list, dtype=[('i', int), ('j', int), ('title_sim', float), ('article_sim', float)])
    np.savez_compressed(output_filename, results=results_array)
    logger.info(f"Pairwise similarity computation complete. Results saved to {output_filename}")
