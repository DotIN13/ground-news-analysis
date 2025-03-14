import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import HashingTF, Tokenizer, MinHashLSH

# Initialize Spark session
spark = SparkSession.builder.appName("Similar Articles") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.executor.memory", "40g") \
        .config("spark.executor.cores", "8") \
        .config("spark.num.executors", "4") \
        .config("spark.executor.memoryOverhead", "4g") \
        .config("spark.driver.memory", "16g") \
        .config("spark.default.parallelism", "256") \
        .config("spark.sql.shuffle.partitions", "256") \
        .getOrCreate()
        
    
# Sample DataFrame (Replace with your data)
df = pd.read_csv("data_random_split.csv")
df = df[["ID", "content"]]
df.rename(columns={"ID": "id"}, inplace=True)
df.dropna(subset=["content"], inplace=True)
df = spark.createDataFrame(df)

df = df.cache()

# Step 1: Tokenize the content
tokenizer = Tokenizer(inputCol="content", outputCol="tokens")
tokenized_df = tokenizer.transform(df)

# Step 2: Convert tokens to feature vectors
hashing_tf = HashingTF(
    inputCol="tokens", 
    outputCol="features", 
    binary=True,  # Use binary frequencies for MinHash
)
featurized_df = hashing_tf.transform(tokenized_df)

# Step 3: Initialize and fit MinHash LSH model
mh_lsh = MinHashLSH(
    inputCol="features",
    outputCol="hashes",
    numHashTables=5  # More tables increase accuracy but require more computation
)
model = mh_lsh.fit(featurized_df)

# Step 4: Identify similar articles
similarity_threshold = 0.3  # Jaccard similarity threshold
similar_articles = model.approxSimilarityJoin(
    featurized_df, 
    featurized_df, 
    threshold=similarity_threshold,
    distCol="jaccardDistance"
)

# Filter out self-comparisons and show results
similar_articles = similar_articles.filter("datasetA.id < datasetB.id")

# Display similar pairs with similarity score
# Use aliases to prevent column name conflict
similar_articles = similar_articles.select(
    col("datasetA.id").alias("id_A"),
    col("datasetB.id").alias("id_B"),
    (1 - col("jaccardDistance")).alias("similarity")
)

similar_articles.write.csv("similar_articles.csv", header=True, mode="overwrite")
