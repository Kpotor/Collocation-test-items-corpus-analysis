# Import required libraries for corpus analysis and statistical calculations
import pandas as pd
import os
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
import AMcalc
from pathlib import Path
import random
import spacy
from spacy.tokens import DocBin
from spacy.matcher import DependencyMatcher

# Configure spaCy to use GPU for faster processing
spacy.prefer_gpu()
spacy_nlp = spacy.load('en_core_web_trf')

# Define file paths for test items and COCA corpus data
TEST_ITEM_LIST_PATH = r"C:\Users\wpoto\Projects\collocation_item_statistics\item_list.csv"
COCA_SPACY_FILES_PATH = r"C:\Users\wpoto\Doctoral_Thesis\Projects\Reference_corpus_data\COCA_spaCy_files"

# Load test items from CSV file
test_items_df = pd.read_csv(TEST_ITEM_LIST_PATH)



# Function to normalize British/American spelling differences
def normalize_spelling(word: str) -> str:
    mapping = {
        "favourite": "favorite",
        "realise": "realize",
        "practise": "practice",
        "judgement": "judgment",
        "homogenous": "homogeneous",
    }
    if not isinstance(word, str):
        return word
    lower = word.lower()
    return mapping.get(lower, lower)


# Extract modifier-noun collocation pairs from test items
modNoun_list = []
for idx, row in test_items_df[test_items_df["rel_type"] == "mod_noun"].iterrows():
    col = (normalize_spelling(row["w1_lemma"]), normalize_spelling(row["w2_lemma"]))
    modNoun_list.append(col)

# Extract verb-object collocation pairs from test items
verbObj_list = []
for idx, row in test_items_df[test_items_df["rel_type"] == "verb_obj"].iterrows():
    col = (normalize_spelling(row["w1_lemma"]), normalize_spelling(row["w2_lemma"]))
    verbObj_list.append(col)

# Create separate lists for individual lemmas (for frequency counting)
modifier_lemma_list = [w1 for w1, w2 in modNoun_list]  # Adjectives/nouns that modify
modified_lemma_list = [w2 for w1, w2 in modNoun_list]  # Nouns being modified

verb_lemma_list = [w1 for w1, w2 in verbObj_list]      # Verbs
obj_lemma_list = [w2 for w1, w2 in verbObj_list]       # Objects


# Define dependency patterns for modifier-noun relationships
# Pattern: noun with adjective or compound modifier
modNoun_rel = [
    {
        "RIGHT_ID": "noun_token",
        "RIGHT_ATTRS": {"POS": "NOUN"}
    },
    {
        "LEFT_ID": "noun_token",
        "REL_OP": ">",
        "RIGHT_ID": "mod_token",
        "RIGHT_ATTRS": {
            "POS": {"IN": ["ADJ", "NOUN"]}, 
            "DEP": {"IN": ["amod", "compound"]}
        }
    }
]

# Define dependency patterns for verb-object relationships
# Pattern: verb with direct object
verbObj_rel = [
    {
        "RIGHT_ID": "verb_token",
        "RIGHT_ATTRS": {"POS": "VERB"}
    },
    {
        "LEFT_ID": "verb_token",
        "REL_OP": ">",
        "RIGHT_ID": "obj_token",
        "RIGHT_ATTRS": {"POS": "NOUN", "DEP": "dobj"}
    }
]

# Initialize dependency matcher and add patterns
matcher = DependencyMatcher(spacy_nlp.vocab)
matcher.add("mod_noun", [modNoun_rel])
matcher.add("verb_obj", [verbObj_rel])


# Function to recursively find all .spacy files in directory tree
def iter_spacy_files(root: str):
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    else:
                        # Find files with .spacy extension (case insensitive)
                        name = entry.name
                        if name.endswith(".spacy") or name.endswith(".SPACY"):
                            yield entry.path
        except (PermissionError, FileNotFoundError):
            continue

# Get all spaCy files from COCA corpus directory
root = COCA_SPACY_FILES_PATH
file_paths = list(iter_spacy_files(root))

# Randomize file order for processing
random.shuffle(file_paths)
print(f"Total files to process: {len(file_paths)}")


# Initialize frequency dictionaries for collocation pairs
modNoun_freq_dict = defaultdict(int)  # Total frequency of modifier-noun pairs
verbObj_freq_dict = defaultdict(int)  # Total frequency of verb-object pairs

# Genre-specific frequency tracking for collocation pairs
modNoun_by_genre = defaultdict(Counter)   # key: (modifier_lemma, noun_lemma) -> genre frequencies
verbObj_by_genre = defaultdict(Counter)   # key: (verb_lemma, object_lemma) -> genre frequencies

# Individual word frequency tracking
modifier_lemma_total_freq = defaultdict(int)  # Frequency of individual modifiers
modified_lemma_total_freq = defaultdict(int)  # Frequency of individual modified nouns
verb_total_freq = defaultdict(int)            # Frequency of individual verbs
obj_total_freq = defaultdict(int)             # Frequency of individual objects

# Global counters for statistical calculations
total_token_count = 0      # Total tokens in corpus
modNoun_total_freq = 0     # Total modifier-noun relationships found
verbObj_total_freq = 0     # Total verb-object relationships found

# Process each spaCy file in the corpus
for file in tqdm(file_paths):
    # Extract genre from filename (format: prefix_genre_suffix)
    file_name = Path(file).stem
    genre = file_name.split("_")[1]
    
    # Load spaCy documents from binary file
    doc_bin = DocBin().from_disk(file)
    docs = list(doc_bin.get_docs(spacy_nlp.vocab))
    
    # Process each document in the file
    for doc in docs:
        total_token_count += len(doc)
        
        # Find dependency matches using the defined patterns
        matches = matcher(doc)
        if matches:
            for match_id, token_ids in matches:
                match_label = spacy_nlp.vocab.strings[match_id]
                
                # Process modifier-noun relationships
                if match_label == "mod_noun":
                    modNoun_total_freq += 1
                    modifier_lemma = normalize_spelling(doc[token_ids[1]].lemma_)
                    modified_lemma = normalize_spelling(doc[token_ids[0]].lemma_)
                    
                    # Count individual word frequencies if they're in our test items
                    if modifier_lemma in modifier_lemma_list:
                        modifier_lemma_total_freq[modifier_lemma] += 1
                    if modified_lemma in modified_lemma_list:
                        modified_lemma_total_freq[modified_lemma] += 1                
                    
                    # Count collocation pair frequency if it's in our test items
                    col = (modifier_lemma, modified_lemma)
                    if col in modNoun_list:
                        modNoun_freq_dict[col] += 1
                        modNoun_by_genre[col][genre] += 1
                        
                # Process verb-object relationships
                elif match_label == "verb_obj":
                    verbObj_total_freq += 1
                    # Extract and normalize lemmas (verb is token_ids[0], object is token_ids[1])
                    verb_lemma = normalize_spelling(doc[token_ids[0]].lemma_)
                    obj_lemma = normalize_spelling(doc[token_ids[1]].lemma_)
                    
                    # Count individual word frequencies if they're in our test items
                    if verb_lemma in verb_lemma_list:
                        verb_total_freq[verb_lemma] += 1
                    if obj_lemma in obj_lemma_list:
                        obj_total_freq[obj_lemma] += 1
                    
                    # Count collocation pair frequency if it's in our test items
                    col = (verb_lemma, obj_lemma)
                    if col in verbObj_list:
                        verbObj_freq_dict[col] += 1
                        verbObj_by_genre[col][genre] += 1

# Print summary statistics
print(f"Total modifier-noun relationships found: {modNoun_total_freq}")
print(f"Total verb-object relationships found: {verbObj_total_freq}")

# Initialize new columns in the dataframe for statistical measures
test_items_df['total_freq'] = 0          # Frequency of the collocation pair
test_items_df['w1_freq'] = 0             # Frequency of first word (modifier/verb)
test_items_df['w2_freq'] = 0             # Frequency of second word (noun/object)
test_items_df['rel_type_total_freq'] = 0 # Total frequency of this relationship type
test_items_df['MI'] = 0.0                # Mutual Information score
test_items_df['t-score'] = 0.0           # T-score measure
test_items_df['z-score'] = 0.0           # Z-score measure
test_items_df['logDice'] = 0.0           # LogDice measure
test_items_df['ΔP(w1->w2)'] = 0.0        # Delta P (w1 -> w2)
test_items_df['ΔP(w2->w1)'] = 0.0        # Delta P (w2 -> w1)

# Initialize genre-specific frequency columns
test_items_df['acad'] = 0    # Academic genre frequency
test_items_df['blog'] = 0    # Blog genre frequency
test_items_df['fic'] = 0     # Fiction genre frequency
test_items_df['mag'] = 0     # Magazine genre frequency
test_items_df['news'] = 0    # News genre frequency
test_items_df['spok'] = 0    # Spoken genre frequency
test_items_df['tvm'] = 0     # TV/Movie genre frequency
test_items_df['web'] = 0     # Web genre frequency


for idx, row in test_items_df.iterrows():
    # Extract and normalize lemmas from the test item
    w1_lemma = row['w1_lemma']
    w2_lemma = row['w2_lemma']
    nw1_lemma = normalize_spelling(w1_lemma)
    nw2_lemma = normalize_spelling(w2_lemma)
    rel_type = row['rel_type']
    
    col_tuple = (nw1_lemma, nw2_lemma)
    
    # Process modifier-noun relationships
    if rel_type == 'mod_noun':
        # Set frequency data for modifier-noun pairs
        test_items_df.at[idx, 'total_freq'] = modNoun_freq_dict.get(col_tuple, 0)
        test_items_df.at[idx, 'rel_type_total_freq'] = modNoun_total_freq
        test_items_df.at[idx, 'w1_freq'] = modifier_lemma_total_freq.get(nw1_lemma, 0)
        test_items_df.at[idx, 'w2_freq'] = modified_lemma_total_freq.get(nw2_lemma, 0)
        
        # Set genre-specific frequencies
        test_items_df.at[idx, 'acad'] = modNoun_by_genre[col_tuple]['acad']
        test_items_df.at[idx, 'blog'] = modNoun_by_genre[col_tuple]['blog']
        test_items_df.at[idx, 'fic'] = modNoun_by_genre[col_tuple]['fic']
        test_items_df.at[idx, 'mag'] = modNoun_by_genre[col_tuple]['mag']
        test_items_df.at[idx, 'news'] = modNoun_by_genre[col_tuple]['news']
        test_items_df.at[idx, 'spok'] = modNoun_by_genre[col_tuple]['spok']
        test_items_df.at[idx, 'tvm'] = modNoun_by_genre[col_tuple]['tvm']
        test_items_df.at[idx, 'web'] = modNoun_by_genre[col_tuple]['web']
        
    # Process verb-object relationships
    elif rel_type == 'verb_obj':
        # Set frequency data for verb-object pairs
        test_items_df.at[idx, 'total_freq'] = verbObj_freq_dict.get(col_tuple, 0)
        test_items_df.at[idx, 'rel_type_total_freq'] = verbObj_total_freq
        test_items_df.at[idx, 'w1_freq'] = verb_total_freq.get(nw1_lemma, 0)
        test_items_df.at[idx, 'w2_freq'] = obj_total_freq.get(nw2_lemma, 0)
        
        # Set genre-specific frequencies
        test_items_df.at[idx, 'acad'] = verbObj_by_genre[col_tuple]['acad']
        test_items_df.at[idx, 'blog'] = verbObj_by_genre[col_tuple]['blog']
        test_items_df.at[idx, 'fic'] = verbObj_by_genre[col_tuple]['fic']
        test_items_df.at[idx, 'mag'] = verbObj_by_genre[col_tuple]['mag']
        test_items_df.at[idx, 'news'] = verbObj_by_genre[col_tuple]['news']
        test_items_df.at[idx, 'spok'] = verbObj_by_genre[col_tuple]['spok']
        test_items_df.at[idx, 'tvm'] = verbObj_by_genre[col_tuple]['tvm']
        test_items_df.at[idx, 'web'] = verbObj_by_genre[col_tuple]['web']
    
    # print(test_items_df.at[idx, 'total_freq'])
    if test_items_df.at[idx, 'total_freq'] > 0:
        AM_dict = AMcalc.AMcalculation(test_items_df.at[idx, 'total_freq'], test_items_df.at[idx, 'w1_freq'], test_items_df.at[idx, 'w2_freq'], test_items_df.at[idx, 'rel_type_total_freq'])
        test_items_df.at[idx, 'MI'] = AM_dict['MI']
        test_items_df.at[idx, 't-score'] = AM_dict['t-score']
        test_items_df.at[idx, 'z-score'] = AM_dict['z-score']
        test_items_df.at[idx, 'logDice'] = AM_dict['logDice']
        test_items_df.at[idx, 'ΔP(w1->w2)'] = AM_dict['ΔP(X->Y)']
        test_items_df.at[idx, 'ΔP(w2->w1)'] = AM_dict['ΔP(Y->X)']
    elif test_items_df.at[idx, 'total_freq'] == 0:
        test_items_df.at[idx, 'MI'] = np.nan
        test_items_df.at[idx, 't-score'] = np.nan
        test_items_df.at[idx, 'z-score'] = np.nan
        test_items_df.at[idx, 'logDice'] = np.nan
        test_items_df.at[idx, 'ΔP(w1->w2)'] = np.nan
        test_items_df.at[idx, 'ΔP(w2->w1)'] = np.nan

# Print final corpus statistics
print(f"COCA total token count: {total_token_count}")

# Save results to Excel file
test_items_df.to_excel(r"item_list_with_statistics_v2.xlsx", index=False)