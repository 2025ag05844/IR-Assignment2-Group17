import math
import random
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st


# ============================================================
# Text Mining and Document Intelligence System
# Assignment implementation:
# - Manual preprocessing, TF-IDF, inverted index and cosine similarity
# - Manual Multinomial Naive Bayes
# - Manual K-Means
# - Manual Purity and Rand Index
# - Content-based Top-K document recommendations
# ============================================================

st.set_page_config(
    page_title="Text Mining and Document Intelligence System",
    page_icon="📚",
    layout="wide",
)

st.title("Text Mining and Document Intelligence System")
st.caption("Information Retrieval Assignment 2 | Explore, analyse and retrieve documents from your text corpus")


# ------------------------- Helpers -------------------------

DEFAULT_STOP_WORDS = set("""
a an the and or but if while with to of in on for from by at as is are was were
be been being have has had do does did this that these those it its he she they
them their his her we you your i me my our ours not no so than then too very can
could should would will just about into over after before between during up down
out off again further once here there when where why how what which who whom
all any both each few more most other some such only own same also back new
said says say get got may might must one two first last now
""".split())


def simple_stem(word):
    """Small educational suffix stemmer; intentionally not a library stemmer."""
    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        base = word[:-3]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("ly"):
        return word[:-2]
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def simple_lemma(word):
    """Lightweight rule-based lemmatization for common English endings."""
    irregular = {
        "children": "child", "men": "man", "women": "woman",
        "people": "person", "mice": "mouse", "better": "good",
        "best": "good", "worse": "bad", "went": "go", "gone": "go",
        "was": "be", "were": "be", "is": "be", "are": "be",
        "has": "have", "had": "have", "did": "do", "done": "do",
    }
    if word in irregular:
        return irregular[word]
    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        base = word[:-3]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def preprocess_text(text, lowercase=True, remove_stopwords=True,
                    stemming=False, lemmatization=False, ngram_max=1):
    if pd.isna(text):
        text = ""
    text = str(text)
    if lowercase:
        text = text.lower()

    # Keep alphabetic tokens and numbers with letters; discard punctuation.
    tokens = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)
    tokens = [token.replace("'", "") for token in tokens if token.strip()]

    if remove_stopwords:
        tokens = [t for t in tokens if t not in DEFAULT_STOP_WORDS]
    if lemmatization:
        tokens = [simple_lemma(t) for t in tokens]
    elif stemming:
        tokens = [simple_stem(t) for t in tokens]

    # N-gram generation: include unigrams, then append consecutive bigrams/trigrams up to max_n.
    max_n = max(1, int(ngram_max))
    if max_n == 1:
        return tokens

    result = list(tokens)
    for n in range(2, max_n + 1):
        result.extend(["_".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
    return result


def manual_tfidf(token_lists):
    """Manual TF-IDF + L2 length normalization. Returns sparse vectors, IDF, index."""
    n_docs = len(token_lists)
    document_frequency = Counter()
    term_counts = []

    for tokens in token_lists:
        counts = Counter(tokens)
        term_counts.append(counts)
        document_frequency.update(counts.keys())

    idf = {
        term: math.log((n_docs + 1) / (df + 1)) + 1
        for term, df in document_frequency.items()
    }

    vectors = []
    for counts in term_counts:
        total_terms = sum(counts.values()) or 1
        weighted = {
            term: (count / total_terms) * idf[term]
            for term, count in counts.items()
        }
        length = math.sqrt(sum(value * value for value in weighted.values()))
        if length:
            weighted = {term: value / length for term, value in weighted.items()}
        vectors.append(weighted)

    inverted_index = defaultdict(list)
    for doc_idx, vector in enumerate(vectors):
        for term, weight in vector.items():
            inverted_index[term].append((doc_idx, weight))

    return vectors, idf, dict(inverted_index), term_counts, document_frequency


def cosine_similarity_sparse(vector_a, vector_b):
    if len(vector_a) > len(vector_b):
        vector_a, vector_b = vector_b, vector_a
    return sum(value * vector_b.get(term, 0.0) for term, value in vector_a.items())


def manual_naive_bayes(train_docs, train_labels, test_docs, alpha=1.0):
    """Manual multinomial Naive Bayes using token counts and Laplace smoothing."""
    class_doc_counts = Counter(train_labels)
    class_token_counts = {label: Counter() for label in class_doc_counts}
    class_total_tokens = Counter()
    vocabulary = set()

    for tokens, label in zip(train_docs, train_labels):
        counts = Counter(tokens)
        class_token_counts[label].update(counts)
        class_total_tokens[label] += sum(counts.values())
        vocabulary.update(counts.keys())

    vocab_size = max(1, len(vocabulary))
    total_docs = len(train_labels)
    predictions = []
    classes = sorted(class_doc_counts)

    for tokens in test_docs:
        token_counts = Counter(token for token in tokens if token in vocabulary)
        best_label, best_score = None, -float("inf")
        for label in classes:
            prior = math.log(class_doc_counts[label] / total_docs)
            denominator = class_total_tokens[label] + alpha * vocab_size
            score = prior
            for token, count in token_counts.items():
                probability = (class_token_counts[label][token] + alpha) / denominator
                score += count * math.log(probability)
            if score > best_score:
                best_label, best_score = label, score
        predictions.append(best_label)
    return predictions, classes


def classification_metrics(actual, predicted, classes):
    correct = sum(a == p for a, p in zip(actual, predicted))
    accuracy = correct / len(actual) if actual else 0.0
    per_class = {}
    for label in classes:
        tp = sum(a == label and p == label for a, p in zip(actual, predicted))
        fp = sum(a != label and p == label for a, p in zip(actual, predicted))
        fn = sum(a == label and p != label for a, p in zip(actual, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"Precision": precision, "Recall": recall, "F1-score": f1}
    macro_precision = sum(x["Precision"] for x in per_class.values()) / max(1, len(per_class))
    macro_recall = sum(x["Recall"] for x in per_class.values()) / max(1, len(per_class))
    macro_f1 = sum(x["F1-score"] for x in per_class.values()) / max(1, len(per_class))
    return {
        "Accuracy": accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1-score": macro_f1,
    }, per_class


def sparse_euclidean_squared(a, b):
    keys = set(a) | set(b)
    return sum((a.get(k, 0.0) - b.get(k, 0.0)) ** 2 for k in keys)


def manual_kmeans(vectors, k=5, max_iterations=20, seed=42):
    """Manual K-Means on sparse TF-IDF vectors; Euclidean distance."""
    if not vectors:
        return [], [], 0
    k = max(1, min(int(k), len(vectors)))
    rng = random.Random(seed)
    initial_indices = rng.sample(range(len(vectors)), k)
    centroids = [dict(vectors[i]) for i in initial_indices]
    assignments = [-1] * len(vectors)

    vector_norms = [sum(value * value for value in vector.values()) for vector in vectors]
    for iteration in range(1, max_iterations + 1):
        changed = False
        new_assignments = []
        centroid_norms = [sum(value * value for value in centroid.values()) for centroid in centroids]
        for doc_idx, vector in enumerate(vectors):
            # Sparse Euclidean distance: ||x||^2 + ||c||^2 - 2(x dot c).
            # Iterating only over the document's terms avoids scanning the full vocabulary.
            distances = []
            for centroid_idx, centroid in enumerate(centroids):
                dot = sum(value * centroid.get(term, 0.0) for term, value in vector.items())
                distance = max(0.0, vector_norms[doc_idx] + centroid_norms[centroid_idx] - 2 * dot)
                distances.append(distance)
            cluster = min(range(k), key=lambda idx: distances[idx])
            new_assignments.append(cluster)
            if cluster != assignments[doc_idx]:
                changed = True

        assignments = new_assignments
        sums = [defaultdict(float) for _ in range(k)]
        sizes = [0] * k
        for vector, cluster in zip(vectors, assignments):
            sizes[cluster] += 1
            for term, value in vector.items():
                sums[cluster][term] += value

        updated_centroids = []
        for cluster in range(k):
            if sizes[cluster] == 0:
                updated_centroids.append(dict(vectors[rng.randrange(len(vectors))]))
            else:
                updated_centroids.append({
                    term: value / sizes[cluster]
                    for term, value in sums[cluster].items()
                    if value != 0
                })
        centroids = updated_centroids
        if not changed:
            return assignments, centroids, iteration

    return assignments, centroids, max_iterations


def manual_purity(true_labels, cluster_labels):
    total = len(true_labels)
    if total == 0:
        return 0.0
    clusters = defaultdict(list)
    for true_label, cluster in zip(true_labels, cluster_labels):
        clusters[cluster].append(true_label)
    correct = sum(Counter(labels).most_common(1)[0][1] for labels in clusters.values())
    return correct / total


def manual_rand_index(true_labels, cluster_labels):
    """Pair-count Rand Index; no external metric implementation."""
    tp = tn = fp = fn = 0
    n = len(true_labels)
    for i in range(n):
        for j in range(i + 1, n):
            same_true = true_labels[i] == true_labels[j]
            same_cluster = cluster_labels[i] == cluster_labels[j]
            if same_true and same_cluster:
                tp += 1
            elif not same_true and not same_cluster:
                tn += 1
            elif not same_true and same_cluster:
                fp += 1
            else:
                fn += 1
    total_pairs = tp + tn + fp + fn
    return (tp + tn) / total_pairs if total_pairs else 0.0


def confusion_matrix_manual(actual, predicted, classes):
    matrix = pd.DataFrame(0, index=classes, columns=classes, dtype=int)
    for a, p in zip(actual, predicted):
        if a in matrix.index and p in matrix.columns:
            matrix.loc[a, p] += 1
    matrix.index.name = "Actual"
    matrix.columns.name = "Predicted"
    return matrix


def plot_confusion_matrix(matrix):
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(matrix.values, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Naive Bayes Confusion Matrix")
    fig.colorbar(image, ax=ax)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix.iloc[i, j]), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    return fig


def prepare_dataset(df, id_col, text_col, category_col):
    result = df.copy()
    result[text_col] = result[text_col].fillna("").astype(str)
    result[category_col] = result[category_col].fillna("Unknown").astype(str)
    result[id_col] = result[id_col].astype(str)
    return result


# ------------------------- Sidebar -------------------------

st.sidebar.header("Data and preprocessing")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV dataset (optional)",
    type=["csv"],
    help="If no file is uploaded, the app tries bbc_news.csv.",
)

default_path = "bbc_news.csv"
try:
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        data_source = uploaded_file.name
    else:
        raw_df = pd.read_csv(default_path)
        data_source = default_path
except FileNotFoundError:
    st.error(
        "Dataset not found. Put your CSV at bbc_news.csv, "
        "or upload it using the sidebar."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()

st.sidebar.caption(f"Loaded: {data_source}")

all_columns = list(raw_df.columns)
if not all_columns:
    st.error("The CSV has no columns.")
    st.stop()


def guess_column(candidates, fallback_index=0):
    lower_map = {str(c).lower(): c for c in all_columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return all_columns[min(fallback_index, len(all_columns) - 1)]


id_default = guess_column(["ArticleId", "ArticleID", "doc_id", "document_id", "id"], 0)
text_default = guess_column(["Text", "text", "content", "article", "body"], min(1, len(all_columns)-1))
category_default = guess_column(["Category", "category", "label", "topic", "class"], min(2, len(all_columns)-1))

id_col = st.sidebar.selectbox("Document ID column", all_columns, index=all_columns.index(id_default))
text_col = st.sidebar.selectbox("Main text column", all_columns, index=all_columns.index(text_default))
category_options = [c for c in all_columns if c != text_col]
if not category_options:
    category_options = all_columns
category_default = category_default if category_default in category_options else category_options[0]
category_col = st.sidebar.selectbox("Category / label column", category_options, index=category_options.index(category_default))

st.sidebar.markdown("---")
st.sidebar.subheader("Preprocessing options")
lowercase = st.sidebar.checkbox("Lowercase text", value=True)
remove_stopwords = st.sidebar.checkbox("Remove stop words", value=True)
stemming = st.sidebar.checkbox("Apply simple stemming", value=False)
lemmatization = st.sidebar.checkbox("Apply simple lemmatization", value=False)
ngram_max = st.sidebar.selectbox("Maximum n-gram size", [1, 2, 3], index=0)
max_docs = st.sidebar.slider(
    "Maximum documents for analytics",
    min_value=min(100, len(raw_df)),
    max_value=max(100, len(raw_df)),
    value=len(raw_df),
    help="Use all documents for final experiments. Lower this only if your laptop is slow.",
)
k_clusters = st.sidebar.slider("Number of K-Means clusters", min_value=2, max_value=8, value=5)
test_fraction = st.sidebar.slider("Naive Bayes test fraction", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
run_button = st.sidebar.button("Run / refresh analysis", type="primary")


# ------------------------- Data ingestion and preparation -------------------------

if text_col == category_col:
    st.error("Choose different columns for the text and category fields.")
    st.stop()

df = prepare_dataset(raw_df, id_col, text_col, category_col)
df = df.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
if len(df) > max_docs:
    df = df.iloc[:max_docs].copy().reset_index(drop=True)

# Cache analytics in session state until user presses the run button or changes controls.
config = (data_source, id_col, text_col, category_col, lowercase, remove_stopwords,
          stemming, lemmatization, ngram_max, max_docs, k_clusters, test_fraction, len(df))
if "analysis_config" not in st.session_state:
    st.session_state.analysis_config = None
if run_button or st.session_state.analysis_config != config:
    with st.spinner("Running manual text-mining algorithms. Larger datasets may take a little time..."):
        original_token_lists = [
            preprocess_text(text, lowercase=False, remove_stopwords=False,
                            stemming=False, lemmatization=False, ngram_max=1)
            for text in df[text_col].tolist()
        ]
        token_lists = [
            preprocess_text(text, lowercase, remove_stopwords, stemming,
                            lemmatization, ngram_max)
            for text in df[text_col].tolist()
        ]
        vectors, idf, inverted_index, term_counts, document_frequency = manual_tfidf(token_lists)

        # Manual train/test split; NB uses token counts and fits vocabulary from training documents only.
        indices_by_class = defaultdict(list)
        for idx, label in enumerate(df[category_col].tolist()):
            indices_by_class[label].append(idx)
        rng = random.Random(42)
        train_indices, test_indices = [], []
        for label_indices in indices_by_class.values():
            shuffled = list(label_indices)
            rng.shuffle(shuffled)
            n_test = max(1, int(len(shuffled) * test_fraction)) if len(shuffled) > 1 else 0
            test_indices.extend(shuffled[:n_test])
            train_indices.extend(shuffled[n_test:])
        if not test_indices and len(df) > 1:
            test_indices = [len(df) - 1]
            train_indices = list(range(len(df) - 1))

        train_docs = [token_lists[i] for i in train_indices]
        train_labels = [df.iloc[i][category_col] for i in train_indices]
        test_docs = [token_lists[i] for i in test_indices]
        actual_labels = [df.iloc[i][category_col] for i in test_indices]
        predicted_labels, classes = manual_naive_bayes(train_docs, train_labels, test_docs)
        metrics, per_class_metrics = classification_metrics(actual_labels, predicted_labels, classes)
        cm = confusion_matrix_manual(actual_labels, predicted_labels, classes)

        cluster_labels, centroids, kmeans_iterations = manual_kmeans(
            vectors, k=k_clusters, max_iterations=20, seed=42
        )
        purity = manual_purity(df[category_col].tolist(), cluster_labels)
        # Pairwise Rand Index is O(n^2); cap only the RI sample for large corpora to keep UI responsive.
        ri_cap = min(1000, len(df))
        # Use a reproducible random sample rather than always taking the first rows.
        ri_sample_indices = sorted(rng.sample(range(len(df)), ri_cap)) if ri_cap else []
        ri_labels = [df.iloc[i][category_col] for i in ri_sample_indices]
        ri_clusters = [cluster_labels[i] for i in ri_sample_indices]
        rand_index = manual_rand_index(ri_labels, ri_clusters)

        st.session_state.analysis = {
            "df": df,
            "original_token_lists": original_token_lists,
            "token_lists": token_lists,
            "vectors": vectors,
            "idf": idf,
            "inverted_index": inverted_index,
            "term_counts": term_counts,
            "document_frequency": document_frequency,
            "train_indices": train_indices,
            "test_indices": test_indices,
            "actual_labels": actual_labels,
            "predicted_labels": predicted_labels,
            "classes": classes,
            "metrics": metrics,
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": cm,
            "cluster_labels": cluster_labels,
            "centroids": centroids,
            "kmeans_iterations": kmeans_iterations,
            "purity": purity,
            "rand_index": rand_index,
            "ri_cap": ri_cap,
            "ri_sample_indices": ri_sample_indices,
        }
        st.session_state.analysis_config = config

analysis = st.session_state.analysis
df = analysis["df"]
token_lists = analysis["token_lists"]
vectors = analysis["vectors"]
cluster_labels = analysis["cluster_labels"]


# ------------------------- Navigation -------------------------

tabs = st.tabs([
    "Dashboard",
    "Preprocessing & TF-IDF",
    "Classification",
    "Clustering",
    "Recommendations",
])


# ------------------------- Dashboard -------------------------

with tabs[0]:
    st.header("Dashboard and Corpus Overview")
    st.success(f"Dataset loaded successfully: {data_source}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", len(df))
    c2.metric("Categories", df[category_col].nunique())
    c3.metric("Vocabulary size", len(analysis["idf"]))
    c4.metric("TF-IDF index terms", len(analysis["inverted_index"]))

    st.subheader("Document preview")
    display_columns = list(dict.fromkeys([id_col, text_col, category_col]))
    st.dataframe(df[display_columns].head(10), use_container_width=True)

    st.subheader("Documents by category")
    st.bar_chart(df[category_col].value_counts())

    st.subheader("Data quality")
    quality = pd.DataFrame({
        "Check": [
            "Missing document IDs",
            "Missing text before filling",
            "Missing category before filling",
            "Duplicate IDs removed",
            "Duplicate exact text rows",
        ],
        "Count": [
            raw_df[id_col].isna().sum(),
            raw_df[text_col].isna().sum(),
            raw_df[category_col].isna().sum(),
            int(raw_df[id_col].duplicated().sum()),
            int(raw_df[text_col].dropna().duplicated().sum()),
        ],
    })
    st.dataframe(quality, use_container_width=True)
    st.info(
        "Exact duplicate text is reported for analysis. Duplicate document IDs are reduced to the first occurrence. "
        "Near-duplicates are not automatically removed; they should be investigated separately."
    )


# ------------------------- Preprocessing and TF-IDF -------------------------

with tabs[1]:
    st.header("Text Mining and Corpus Analytics Hub")
    st.write(
        "Explore the document collection, inspect preprocessing, compare corpus statistics before and after "
        "cleaning, and examine the manually generated TF-IDF features and inverted index."
    )

    original_token_lists = analysis["original_token_lists"]
    processed_token_lists = analysis["token_lists"]

    # Corpus-level document length statistics.
    raw_lengths = df[text_col].fillna("").astype(str).str.split().str.len()
    before_lengths = [len(tokens) for tokens in original_token_lists]
    after_lengths = [len(tokens) for tokens in processed_token_lists]
    before_vocab = set(token for tokens in original_token_lists for token in tokens)
    after_vocab = set(token for tokens in processed_token_lists for token in tokens)
    before_total = sum(before_lengths)
    after_total = sum(after_lengths)

    st.subheader("1. Corpus overview")
    overview_cols = st.columns(4)
    overview_cols[0].metric("Documents", f"{len(df):,}")
    overview_cols[1].metric("Categories / labels", f"{df[category_col].nunique():,}")
    overview_cols[2].metric("Total words in source text", f"{int(raw_lengths.sum()):,}")
    overview_cols[3].metric("Average source words / document", f"{raw_lengths.mean():,.1f}" if len(raw_lengths) else "0")

    length_stats = pd.DataFrame(
        [
            {
                "Stage": "Source text (whitespace word count)",
                "Average length": round(float(raw_lengths.mean()), 2) if len(raw_lengths) else 0,
                "Minimum length": int(raw_lengths.min()) if len(raw_lengths) else 0,
                "Maximum length": int(raw_lengths.max()) if len(raw_lengths) else 0,
                "Total words/tokens": int(raw_lengths.sum()),
            },
            {
                "Stage": "Baseline tokens (punctuation removed)",
                "Average length": round(sum(before_lengths) / max(1, len(before_lengths)), 2),
                "Minimum length": min(before_lengths, default=0),
                "Maximum length": max(before_lengths, default=0),
                "Total words/tokens": before_total,
            },
            {
                "Stage": "After selected preprocessing",
                "Average length": round(sum(after_lengths) / max(1, len(after_lengths)), 2),
                "Minimum length": min(after_lengths, default=0),
                "Maximum length": max(after_lengths, default=0),
                "Total words/tokens": after_total,
            },
        ]
    )
    st.markdown("**Document length statistics**")
    st.dataframe(length_stats, use_container_width=True, hide_index=True)

    st.markdown("**Documents by category / label**")
    category_counts = df[category_col].value_counts().rename_axis("Category / label").reset_index(name="Document count")
    category_counts["Percentage"] = (category_counts["Document count"] / max(1, len(df)) * 100).round(2)
    left, right = st.columns([1, 1])
    with left:
        st.dataframe(category_counts, use_container_width=True, hide_index=True)
    with right:
        st.bar_chart(category_counts.set_index("Category / label")["Document count"])

    st.divider()
    st.subheader("2. Before-and-after preprocessing comparison")
    st.caption(
        "The baseline is tokenized with punctuation removed, but without the selected stop-word, stemming, "
        "lemmatization or n-gram settings. It is not a byte-for-byte raw-text measurement."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vocabulary before", f"{len(before_vocab):,}")
    c2.metric("Vocabulary after", f"{len(after_vocab):,}", delta=f"{len(after_vocab) - len(before_vocab):,}")
    c3.metric("Tokens before", f"{before_total:,}")
    c4.metric("Tokens after", f"{after_total:,}", delta=f"{after_total - before_total:,}")

    comparison_df = pd.DataFrame(
        [
            {"Measure": "Vocabulary size", "Before": len(before_vocab), "After": len(after_vocab)},
            {"Measure": "Total tokens", "Before": before_total, "After": after_total},
            {"Measure": "Average tokens per document", "Before": round(before_total / max(1, len(df)), 2),
             "After": round(after_total / max(1, len(df)), 2)},
        ]
    )
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    before_counts = Counter(token for tokens in original_token_lists for token in tokens)
    after_counts = Counter(token for tokens in processed_token_lists for token in tokens)

    # Vocabulary and term-frequency summary statistics.
    def frequency_summary(counts):
        frequencies = list(counts.values())
        return {
            "Unique terms (vocabulary)": len(counts),
            "Total term occurrences": sum(frequencies),
            "Mean term frequency": round(sum(frequencies) / len(frequencies), 2) if frequencies else 0,
            "Median term frequency": float(pd.Series(frequencies).median()) if frequencies else 0,
            "Minimum term frequency": min(frequencies, default=0),
            "Maximum term frequency": max(frequencies, default=0),
            "Hapax legomena (frequency = 1)": sum(1 for value in frequencies if value == 1),
            "Terms occurring <= 2 times": sum(1 for value in frequencies if value <= 2),
        }

    before_summary = frequency_summary(before_counts)
    after_summary = frequency_summary(after_counts)
    summary_df = pd.DataFrame({
        "Statistic": list(before_summary.keys()),
        "Before preprocessing": list(before_summary.values()),
        "After preprocessing": list(after_summary.values()),
    })
    st.markdown("**Vocabulary and term-frequency summary statistics**")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    reduction = (
        (len(before_vocab) - len(after_vocab)) / len(before_vocab) * 100
        if before_vocab else 0
    )
    st.metric("Vocabulary size change", f"{reduction:.2f}% reduction",
              help="Calculated relative to the baseline vocabulary. A negative value means vocabulary increased.")

    st.markdown("**Most frequent terms: before vs after preprocessing**")
    freq_left, freq_right = st.columns(2)
    with freq_left:
        st.caption("Before selected preprocessing")
        if before_counts:
            st.bar_chart(pd.Series(dict(before_counts.most_common(20)), name="Term frequency"))
        else:
            st.info("No tokens available before preprocessing.")
    with freq_right:
        st.caption("After selected preprocessing")
        if after_counts:
            st.bar_chart(pd.Series(dict(after_counts.most_common(20)), name="Term frequency")
                         )
        else:
            st.info("No tokens remain after the selected preprocessing.")

    st.markdown("**Full vocabulary term-frequency distributions**")
    st.caption(
        "Each histogram shows how many vocabulary terms occur at a given frequency. "
        "The logarithmic x-axis helps display both common and rare terms."
    )
    hist_left, hist_right = st.columns(2)
    for target_col, counts, chart_title in [
        (hist_left, before_counts, "Before preprocessing"),
        (hist_right, after_counts, "After preprocessing"),
    ]:
        with target_col:
            st.caption(chart_title)
            if counts:
                values = list(counts.values())
                fig, ax = plt.subplots(figsize=(7, 4))
                max_frequency = max(values)
                if max_frequency > 1:
                    bins = sorted(set([1] + [max(2, int(x)) for x in
                        np.logspace(0, math.log10(max_frequency + 1), 18)]))
                    ax.hist(values, bins=bins, edgecolor="black")
                    ax.set_xscale("log")
                else:
                    ax.hist(values, bins=5, edgecolor="black")
                ax.set_xlabel("Term frequency (log scale when applicable)")
                ax.set_ylabel("Number of unique terms")
                ax.set_title(f"Vocabulary frequency distribution — {chart_title}")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No term frequencies to plot.")

    st.markdown("**Rank-frequency (Zipf-style) distribution**")
    st.caption("Terms are ranked from most frequent to least frequent. This helps reveal whether a small number of terms dominate the corpus.")
    rank_left, rank_right = st.columns(2)
    for target_col, counts, chart_title in [
        (rank_left, before_counts, "Before preprocessing"),
        (rank_right, after_counts, "After preprocessing"),
    ]:
        with target_col:
            if counts:
                ranked = sorted(counts.values(), reverse=True)
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(range(1, len(ranked) + 1), ranked)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("Term rank (log scale)")
                ax.set_ylabel("Term frequency (log scale)")
                ax.set_title(chart_title)
                ax.grid(True, alpha=0.25)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No terms available for this distribution.")

    # Document frequency indicates how many distinct documents contain each term.
    st.markdown("**Document-frequency distribution after preprocessing**")
    processed_df = Counter()
    for tokens in processed_token_lists:
        processed_df.update(set(tokens))
    if processed_df:
        df_values = list(processed_df.values())
        df_stats = pd.DataFrame([{
            "Unique terms": len(processed_df),
            "Mean documents per term": round(sum(df_values) / len(df_values), 2),
            "Median documents per term": float(pd.Series(df_values).median()),
            "Minimum document frequency": min(df_values),
            "Maximum document frequency": max(df_values),
            "Terms in one document only": sum(v == 1 for v in df_values),
        }])
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
        top_df_terms = pd.Series(dict(Counter(processed_df).most_common(20)), name="Document frequency")
        st.bar_chart(top_df_terms)
    else:
        st.info("No document-frequency features are available after preprocessing.")

    st.markdown("**Document-length distributions**")
    length_left, length_right = st.columns(2)
    for target_col, lengths, chart_title in [
        (length_left, before_lengths, "Before preprocessing"),
        (length_right, after_lengths, "After preprocessing"),
    ]:
        with target_col:
            fig, ax = plt.subplots(figsize=(7, 4))
            if lengths:
                ax.hist(lengths, bins=30, edgecolor="black")
            ax.set_xlabel("Tokens per document")
            ax.set_ylabel("Number of documents")
            ax.set_title(chart_title)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


    st.subheader("Inspect an individual document")
    sample_idx = st.number_input(
        "Document row index", min_value=0, max_value=max(0, len(df) - 1), value=0, step=1
    )
    selected_row = df.iloc[int(sample_idx)]
    st.caption(
        f"Document ID: {selected_row[id_col]} | Category / label: {selected_row[category_col]}"
    )
    before_col, after_col = st.columns(2)
    with before_col:
        st.markdown("**Source text**")
        st.text(str(selected_row[text_col])[:2500])
        st.markdown("**Baseline tokens**")
        st.write(original_token_lists[int(sample_idx)][:150])
    with after_col:
        st.markdown("**Processed tokens**")
        st.write(processed_token_lists[int(sample_idx)][:250])

    st.divider()
    st.subheader("3. N-gram generation")
    st.write(
        "N-grams are sequences of consecutive tokens. A unigram contains one token, a bigram contains two "
        "consecutive tokens, and a trigram contains three consecutive tokens. The sidebar setting controls the "
        "maximum n-gram size generated for the current analysis."
    )
    ngram_counts = Counter()
    for tokens in processed_token_lists:
        for token in tokens:
            ngram_counts[token.count("_") + 1] += 1
    ng1, ng2, ng3 = st.columns(3)
    ng1.metric("Unigram features", f"{ngram_counts.get(1, 0):,}")
    ng2.metric("Bigram features", f"{ngram_counts.get(2, 0):,}")
    ng3.metric("Trigram features", f"{ngram_counts.get(3, 0):,}")
    st.caption(
        f"Maximum n-gram size currently selected: {ngram_max}. "
        "For example, the phrase 'machine learning model' produces the bigrams "
        "'machine_learning' and 'learning_model', and the trigram 'machine_learning_model'."
    )
    if ngram_max == 1:
        st.info("Only unigrams are currently generated. Select 2 or 3 in the sidebar to include bigrams or trigrams.")
    else:
        example_tokens = processed_token_lists[int(sample_idx)]
        ngram_examples = pd.DataFrame(
            [{"N-gram": token, "N": token.count("_") + 1}
             for token in example_tokens if token.count("_") + 1 > 1][:30]
        )
        if not ngram_examples.empty:
            st.markdown("**N-gram examples from the selected document**")
            st.dataframe(ngram_examples, use_container_width=True, hide_index=True)
        else:
            st.info("No multi-word n-grams were generated for this document. Try another document.")

    st.divider()
    st.subheader("4. Manual TF-IDF feature analysis")
    st.write(
        "**TF-IDF** gives each term a weight within a document. Term Frequency (TF) reflects how often a term "
        "appears in that document, while Inverse Document Frequency (IDF) gives less weight to terms appearing "
        "in many documents and relatively more weight to terms that are more distinctive across the corpus."
    )
    st.code(
        "TF(t, d) = count(t in d) / total tokens in d\n"
        "IDF(t) = log((N + 1) / (DF(t) + 1)) + 1\n"
        "TF-IDF(t, d) = TF(t, d) × IDF(t)",
        language="text",
    )
    st.write(
        "The app calculates these values manually. It then applies L2 length normalization to each document "
        "vector so that vector magnitude is not driven simply by document length."
    )
    st.code(
        "L2 norm = sqrt(sum(weight(t)^2 for every term t in the document))\n"
        "normalized_weight(t) = TF-IDF(t) / L2 norm",
        language="text",
    )
    tfidf_cols = st.columns(3)
    tfidf_cols[0].metric("Vocabulary / indexed terms", f"{len(analysis['idf']):,}")
    tfidf_cols[1].metric("Documents with vectors", f"{len(analysis['vectors']):,}")
    tfidf_cols[2].metric("Total index postings", f"{sum(len(x) for x in analysis['inverted_index'].values()):,}")

    st.divider()
    st.subheader("5. Inverted index explorer")
    st.write(
        "An inverted index maps each term to the documents that contain it. Each posting below identifies a "
        "document and its L2-normalized TF-IDF weight for the selected term. This structure helps retrieve "
        "documents containing a query term without scanning every document from scratch."
    )
    indexed_terms = sorted(analysis["inverted_index"].keys())
    if indexed_terms:
        chosen_term = st.selectbox("Inspect an indexed term", indexed_terms[:5000])
        if chosen_term:
            postings = analysis["inverted_index"].get(chosen_term, [])
            posting_df = pd.DataFrame([
                {
                    "Document ID": df.iloc[doc_i][id_col],
                    "Category / label": df.iloc[doc_i][category_col],
                    "TF-IDF weight (L2 normalized)": round(weight, 6),
                }
                for doc_i, weight in postings[:100]
            ])
            st.write(f"IDF for **{chosen_term}**: {analysis['idf'].get(chosen_term, 0):.6f}")
            st.write(f"Document frequency: {len(postings):,} | Showing up to 100 postings.")
            st.dataframe(posting_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No index terms were generated. Check whether the selected text column contains usable text.")

    st.info(
        "Implementation note: TF-IDF, L2 length normalization and the inverted index are implemented manually "
        "in this app; no ready-made TF-IDF vectorizer is used. Stemming and lemmatization are simplified "
        "rule-based educational implementations, not full linguistic NLP models."
    )


# ------------------------- Classification -------------------------

with tabs[2]:
    st.header("Document Classification — Manual Naive Bayes")
    st.write(
        "A stratified-by-category random split is used. The classifier learns word counts and class priors from the "
        "training documents, then predicts labels for held-out test documents. Naive Bayes is implemented manually."
    )
    metrics = analysis["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
    c2.metric("Macro precision", f"{metrics['Macro Precision']:.3f}")
    c3.metric("Macro recall", f"{metrics['Macro Recall']:.3f}")
    c4.metric("Macro F1-score", f"{metrics['Macro F1-score']:.3f}")

    st.subheader("Actual vs predicted category distribution")
    dist = pd.DataFrame({
        "Actual": pd.Series(analysis["actual_labels"]).value_counts(),
        "Predicted": pd.Series(analysis["predicted_labels"]).value_counts(),
    }).fillna(0).astype(int)
    st.bar_chart(dist)

    st.subheader("Confusion matrix")
    st.pyplot(plot_confusion_matrix(analysis["confusion_matrix"]))
    st.dataframe(analysis["confusion_matrix"], use_container_width=True)

    st.subheader("Per-class metrics")
    per_class_df = pd.DataFrame(analysis["per_class_metrics"]).T
    st.dataframe(per_class_df.round(3), use_container_width=True)

    st.subheader("Actual and predicted labels")
    result_rows = []
    for idx, actual, predicted in zip(
        analysis["test_indices"], analysis["actual_labels"], analysis["predicted_labels"]
    ):
        result_rows.append({
            "Document ID": df.iloc[idx][id_col],
            "True category": actual,
            "Predicted category": predicted,
            "Correct": actual == predicted,
            "Text preview": str(df.iloc[idx][text_col])[:180],
        })
    st.dataframe(pd.DataFrame(result_rows), use_container_width=True)


# ------------------------- Clustering -------------------------

with tabs[3]:
    st.header("Document Clustering — Manual K-Means")
    st.write(
        "K-Means initializes centroids from selected document vectors, assigns each document to the nearest centroid "
        "using Euclidean distance, updates centroids as cluster means, and stops when assignments stop changing or "
        "20 iterations are reached."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters requested", k_clusters)
    c2.metric("Iterations completed", analysis["kmeans_iterations"])
    c3.metric("Purity", f"{analysis['purity']:.3f}")

    st.subheader("Cluster sizes")
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    cluster_counts.index = [f"Cluster {i}" for i in cluster_counts.index]
    st.bar_chart(cluster_counts)

    st.subheader("2D cluster visualization")
    # Use two frequent TF-IDF terms as a transparent 2D feature projection for plotting only.
    top_terms = [term for term, _ in Counter(
        term for vector in vectors for term in vector.keys()
    ).most_common(2)]
    if len(top_terms) < 2:
        top_terms = (top_terms + ["_no_second_term"])[:2]
    plot_df = pd.DataFrame({
        "x": [v.get(top_terms[0], 0.0) for v in vectors],
        "y": [v.get(top_terms[1], 0.0) for v in vectors],
        "Cluster": [f"Cluster {c}" for c in cluster_labels],
    })
    fig, ax = plt.subplots(figsize=(8, 5))
    for cluster_name, group in plot_df.groupby("Cluster"):
        ax.scatter(group["x"], group["y"], label=cluster_name, alpha=0.65, s=18)
    ax.set_xlabel(f"TF-IDF weight: {top_terms[0]}")
    ax.set_ylabel(f"TF-IDF weight: {top_terms[1]}")
    ax.set_title("Document clusters projected onto two term dimensions")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(
        "For this view only, each document is projected onto two frequent term dimensions. "
        "The clustering itself uses the full sparse TF-IDF vectors."
    )

    st.subheader("Clustering evaluation")
    c1, c2 = st.columns(2)
    c1.metric("Purity (manual)", f"{analysis['purity']:.4f}")
    c2.metric("Rand Index (manual)", f"{analysis['rand_index']:.4f}")
    if analysis["ri_cap"] < len(df):
        st.caption(
            f"Rand Index was computed on the first {analysis['ri_cap']} documents to keep pairwise evaluation responsive; "
            f"Purity uses all {len(df)} documents."
        )

    st.subheader("Cluster membership")
    membership = pd.DataFrame({
        "Document ID": df[id_col],
        "True category": df[category_col],
        "Assigned cluster": [f"Cluster {c}" for c in cluster_labels],
        "Text preview": df[text_col].astype(str).str.slice(0, 160),
    })
    st.dataframe(membership, use_container_width=True)

    st.subheader("Classification and clustering comparison")
    comparison = pd.DataFrame([
        {"Method": "Naive Bayes classification", "Metric": "Accuracy", "Value": analysis["metrics"]["Accuracy"]},
        {"Method": "Naive Bayes classification", "Metric": "Macro F1-score", "Value": analysis["metrics"]["Macro F1-score"]},
        {"Method": "K-Means clustering", "Metric": "Purity", "Value": analysis["purity"]},
        {"Method": "K-Means clustering", "Metric": "Rand Index", "Value": analysis["rand_index"]},
    ])
    st.dataframe(comparison, use_container_width=True)
    st.info(
        "Naive Bayes is supervised: it learns from category labels. K-Means is unsupervised: it groups documents "
        "by vector similarity without using labels during cluster assignment. Their metrics therefore measure different things."
    )


# ------------------------- Recommendations -------------------------

with tabs[4]:
    st.header("Content-Based Document Recommendation")
    st.write(
        "Choose a document and retrieve the Top-K documents with the highest cosine similarity between their "
        "manually calculated, length-normalized TF-IDF vectors. The selected document itself is excluded."
    )
    doc_options = [
        f"{df.iloc[i][id_col]} | {str(df.iloc[i][text_col])[:90]}"
        for i in range(len(df))
    ]
    selected_option = st.selectbox("Choose a document", doc_options)
    selected_idx = doc_options.index(selected_option)
    top_k = st.slider("Number of recommendations (K)", min_value=1, max_value=10, value=5)

    if st.button("Find similar documents"):
        selected_vector = vectors[selected_idx]
        scores = []
        for candidate_idx, candidate_vector in enumerate(vectors):
            if candidate_idx == selected_idx:
                continue
            score = cosine_similarity_sparse(selected_vector, candidate_vector)
            scores.append((candidate_idx, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        rec_rows = []
        for candidate_idx, score in scores[:top_k]:
            rec_rows.append({
                "Document ID": df.iloc[candidate_idx][id_col],
                "Category": df.iloc[candidate_idx][category_col],
                "Cosine similarity": round(score, 4),
                "Text preview": str(df.iloc[candidate_idx][text_col])[:350],
            })
        st.subheader("Top recommended documents")
        recommendations_df = pd.DataFrame(rec_rows)
        st.dataframe(recommendations_df, use_container_width=True, hide_index=True)
        if not recommendations_df.empty:
            chart_data = recommendations_df.set_index("Document ID")["Cosine similarity"].sort_values()
            st.markdown("**Recommendation similarity scores**")
            st.bar_chart(chart_data)

    st.markdown("**How it works:**")
    st.write(
        "Each article is represented by a manually calculated TF-IDF vector. Cosine similarity compares the direction "
        "of two vectors; higher scores indicate more similar term usage."
    )
    st.markdown("**Limitations:**")
    st.write(
        "Content-based recommendations depend on article text and vocabulary. They may miss semantic relationships "
        "expressed with different words and do not personalize recommendations to a user. Collaborative recommendation "
        "would require user–document interaction data, which this news dataset does not provide."
    )


# ------------------------- Footer / guidance -------------------------

st.divider()
st.caption(
    "Implementation note: TF-IDF, cosine similarity, Naive Bayes, K-Means, Purity, and Rand Index are implemented "
    "manually in this application. Simple rule-based stemming and lemmatization are educational approximations, "
    "not full linguistic NLP tools."
)
