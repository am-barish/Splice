import config
from transformers import AutoTokenizer, AutoModel
from datasets_splice import *
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

import os

device = torch.device("mps")


LSA_DIM = 150

lsa_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=10,
        max_df=0.85,
        sublinear_tf=True,
        strip_accents='unicode'
    )),
    ('svd', TruncatedSVD(n_components=LSA_DIM, random_state=42))
])

def extract_features_lsa(texts, fit=False):
    global lsa_pipeline
    if fit:
        print("Fitting LSA (TF-IDF + SVD) on all texts...")
        return lsa_pipeline.fit_transform(texts)

    print("Transforming texts...")
    return lsa_pipeline.transform(texts)

def extract_features(texts, batch_size=128):
    features = []
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt',
                          padding=True, truncation=True, max_length=256)

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            feat = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            features.append(feat)

        print(f"Processed {i+batch_size}/{len(texts)}", end='\r')

    print()
    return np.vstack(features)

def extract_features_tfidf(texts, fit=False):
    global tfidf_vectorizer
    if fit:
        print("Fitting TF-IDF on all texts...")
        tfidf_vectorizer.fit(texts)

    print("Transforming texts with TF-IDF...")
    features = tfidf_vectorizer.transform(texts).toarray()
    return features

def get_amazon_info():
    lsa_cache = f'reviews_features_lsa{LSA_DIM}.parquet'

    if os.path.exists(lsa_cache):
        print(f"Loading cached LSA-{LSA_DIM} features from {lsa_cache}...")
        df = pd.read_parquet(lsa_cache)
    else:
        df = amazon_review_setup()

        print(f"Fitting LSA (TF-IDF + SVD to {LSA_DIM} dims) on all {len(df)} reviews...")
        texts = df['review_body'].fillna('').tolist()
        feats_array = extract_features_lsa(texts, fit=True)

        print(f"LSA features shape: {feats_array.shape}")
        df['features'] = list(feats_array)
        df.to_parquet(lsa_cache)
        print(f"Saved to {lsa_cache}")

    if 'split' not in df.columns:
        print("Adding split column...")
        df["split"] = "train"
        for cat in ['Movies_and_TV', 'Magazine_Subscriptions', 'Handmade_Products']:
            cat_df = df[df["category"] == cat]
            idx = cat_df.index.to_numpy()
            rng = np.random.default_rng(42)
            rng.shuffle(idx)
            n_val = int(0.1 * len(idx))
            df.loc[idx[:n_val], "split"] = "val"


    sources = {}
    for cat in df['category'].unique():
        cat_df = df[df['category'] == cat]
        feats = np.stack(cat_df['features'].values)
        labels = cat_df['label'].values
        sources[cat] = (feats, labels, cat_df["split"].values)

    source_cats = [cat for cat in sources
                   if cat not in ['Movies_and_TV', 'Magazine_Subscriptions', 'Handmade_Products']]
    target_cats = ['Movies_and_TV', 'Magazine_Subscriptions', 'Handmade_Products']
    source_sources = {cat: sources[cat] for cat in source_cats}
    target_sources = {cat: sources[cat] for cat in target_cats}

    return source_cats, target_cats, source_sources, target_sources


def soft_f1_loss_binary(logits, labels, eps=1e-7):
    if labels.dim() == 1:
        labels = labels.unsqueeze(1)
    probs = torch.sigmoid(logits)

    tp = (probs * labels).sum()
    fp = (probs * (1 - labels)).sum()
    fn = ((1 - probs) * labels).sum()

    f1 = (2 * tp) / (2 * tp + fp + fn + eps)
    return 1.0 - f1

def build_head_binary(feat_dim=None, device='cpu'):
    if feat_dim is None:
        feat_dim = LSA_DIM
    head = nn.Sequential(
        nn.LayerNorm(feat_dim),
        nn.Linear(feat_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1)
    ).to(device)
    return head

def train_binary(head, feats_list, labels_list, epochs=15, lr=1e-3, weight_balanced=True):
    device = next(head.parameters()).device

    X_train = torch.tensor(np.vstack(feats_list)).float().to(device)
    y_train = torch.tensor(np.hstack(labels_list)).float().unsqueeze(1).to(device)

    if weight_balanced:
        pos_weight = torch.tensor([(y_train==0).sum() / (y_train==1).sum()]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)

    head.train()
    for epoch in range(epochs):
        logits = head(X_train)
        loss = criterion(logits, y_train)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        scheduler.step(loss)

        if epoch % 10 == 0:
            prob = torch.sigmoid(logits)
            acc = ((prob > 0.5).float() == y_train).float().mean().item()

    return head


def evaluate_binary(head, test_sources, threshold=0.5):
    head.eval()
    f1s = []
    device = next(head.parameters()).device

    with torch.no_grad():
        for cat, (feats, labs, splits) in test_sources.items():
            test_mask = splits == "train"
            X_test = torch.tensor(feats[test_mask]).float().to(device)
            y_test = torch.tensor(labs[test_mask]).cpu().numpy()

            logits = head(X_test)
            prob = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            pred = (prob > threshold).astype(int)

            f1 = f1_score(y_test, pred, average='binary')
            f1s.append(f1)

    return float(np.mean(f1s))


def get_subset_f1_amazon(subset_indices, ctx):
    return get_subset_score(subset_indices, ctx)

def get_subset_score(subset, ctx):
    torch.manual_seed(42)
    np.random.seed(42)


    idx2cats = {i: ctx.amazon_source_cats[i] for i in range(len(ctx.amazon_source_cats))}
    chosen_cats = [idx2cats[i] for i in subset]

    feats_list  = [ctx.source_pool["Amazon"][cat]["feats"].numpy()  for cat in chosen_cats]
    labels_list = [ctx.source_pool["Amazon"][cat]["labels"].numpy() for cat in chosen_cats]

    head = build_head_binary(device='cpu')
    train_binary(head, feats_list, labels_list, epochs=15)
    subset_f1 = evaluate_binary(head, ctx.amazon_target_sources)
    return subset_f1


def get_target_split(target_sources):
    print(target_sources.keys())
    movies_feats, movies_labels, movies_splits = target_sources['Movies_and_TV']
    magazine_feats, magazine_labels, magazine_splits = target_sources['Magazine_Subscriptions']
    handmade_feats, handmade_labels, handmade_splits = target_sources['Handmade_Products']

    target_feats = np.vstack([movies_feats, magazine_feats, handmade_feats])
    target_labels = np.hstack([movies_labels, magazine_labels, handmade_labels])
    target_splits = np.hstack([movies_splits, magazine_splits, handmade_splits])

    val_mask = target_splits == "val"
    test_mask = target_splits == "train"

    target_val_feats = torch.tensor(target_feats[val_mask])
    target_val_labels = torch.tensor(target_labels[val_mask])
    target_test_feats = torch.tensor(target_feats[test_mask])
    target_test_labels = torch.tensor(target_labels[test_mask])

    return target_val_feats, target_val_labels, target_test_feats, target_test_labels

def train_binary_soft_f1(head, feats_list, labels_list, epochs=50, lr=1e-3):
    device = next(head.parameters()).device

    X_train = torch.tensor(np.vstack(feats_list)).float().to(device)
    y_train = torch.tensor(np.hstack(labels_list)).float().unsqueeze(1).to(device)

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)

    head.train()
    for epoch in range(epochs):
        logits = head(X_train)
        loss = soft_f1_loss_binary(logits, y_train)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        scheduler.step(loss)

    return head

def warm_source_head(source_sources, epochs=30):
    train_feats_list, train_labels_list = [], []
    for cat, (feats, labels, splits) in source_sources.items():
        train_mask = splits == "train"
        train_feats_list.append(feats[train_mask])
        train_labels_list.append(labels[train_mask])

    head = build_head_binary(device='cpu')
    train_binary(head, train_feats_list, train_labels_list, epochs=epochs)
    train_binary_soft_f1(head, train_feats_list, train_labels_list, epochs=30)
    return head


def safe_cosine_sim(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    min_len = min(len(a_flat), len(b_flat))
    a_trunc = F.normalize(a_flat[:min_len], p=2, dim=0)
    b_trunc = F.normalize(b_flat[:min_len], p=2, dim=0)
    return F.cosine_similarity(a_trunc.unsqueeze(0), b_trunc.unsqueeze(0)).item()

def alignment_splice(g_T, source_grads, K=5):
    alignments = {}
    for cat, g_src in source_grads.items():
        alignments[cat] = safe_cosine_sim(g_T, g_src)

    top_cats = sorted(alignments, key=alignments.get, reverse=True)[:K]
    print("Top alignments:", [(cat, f"{alignments[cat]:.3f}") for cat in top_cats])
    return top_cats

def compute_gradients_normalized(head, sources_dict, subset_size=512):
    head.eval()
    grads_norm = {}

    for name, (feats, labels) in sources_dict.items():
        idx = np.random.choice(len(feats), min(subset_size, len(feats)), replace=False)
        X = torch.tensor(feats[idx]).float().to(device)
        y = torch.tensor(labels[idx]).float().unsqueeze(1).to(device)

        logits = head(X)
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction='none')

        head.zero_grad()
        loss.mean().backward()

        grad = head[1].weight.grad.flatten().detach().cpu()
        grad_norm = F.normalize(grad, dim=0)
        grads_norm[name] = grad_norm

        print(f"{name}: grad_norm={grad.norm().item():.3f}")

    return grads_norm

def stratified_sample(feats, labels, subset_size):
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    pos_idx = np.where(labels_np == 1)[0]
    neg_idx = np.where(labels_np == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        n = min(subset_size, len(labels_np))
        return torch.randperm(len(labels_np))[:n]

    half = subset_size // 2
    n_pos = min(half, len(pos_idx))
    n_neg = min(subset_size - n_pos, len(neg_idx))

    chosen_pos = np.random.choice(pos_idx, n_pos, replace=False)
    chosen_neg = np.random.choice(neg_idx, n_neg, replace=False)
    chosen = np.concatenate([chosen_pos, chosen_neg])
    np.random.shuffle(chosen)
    return torch.tensor(chosen, dtype=torch.long)

def compute_target_gradient(feats, labels, head, device, subset_size=512):
    head.train()
    head.zero_grad()
    n = min(subset_size, len(feats))
    idx = stratified_sample(feats, labels, subset_size)
    X = feats[idx].to(torch.float32).to(device)
    y = labels[idx].to(device).float()

    logits = head(X)
    loss = soft_f1_loss_binary(logits, y)
    loss.backward()


    final_layer = None
    for layer in head:
        if isinstance(layer, nn.Linear):
            final_layer = layer
    grad = final_layer.weight.grad.flatten().detach().cpu()

    return F.normalize(grad, dim=0)

def compute_location_gradient(feats, labels, head, device, subset_size=512):
    return compute_target_gradient(feats, labels, head, device, subset_size)

def compute_target_gradient_averaged(feats, labels, head, device,
                                     subset_size=512, n_seeds=5):
    grads = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        g = compute_target_gradient(feats, labels, head, device, subset_size)
        grads.append(g)

    g_mean = torch.stack(grads).mean(dim=0)
    return F.normalize(g_mean, dim=0)

def amazon_grad_match(source_sources, target_sources, source_cats):
    print("=== AMAZON GRAD MATCH ===")
    grad_device = torch.device('cpu')

    warm_head = warm_source_head(source_sources)

    target_val_feats, target_val_labels, target_test_feats, target_test_labels = \
        get_target_split(target_sources)

    g_T = compute_target_gradient_averaged(
        target_val_feats, target_val_labels, warm_head, grad_device,
        subset_size=512, n_seeds=5
    )

    source_train_grads = {}
    for cat, (feats, labels, splits) in source_sources.items():
        train_mask = splits == "train"
        if train_mask.sum() < 32:
            continue
        source_train_feats = torch.tensor(feats[train_mask])
        source_train_labels = torch.tensor(labels[train_mask])
        g_cat = compute_location_gradient(
            source_train_feats, source_train_labels, warm_head, grad_device
        )
        source_train_grads[cat] = g_cat

    return source_train_grads, g_T

def validate_gradient_proxy(
    source_sources, target_sources, source_cats,
    g_T, cat_grads,
    n_random_subsets=20, subset_size=5
):

    n_sources = len(source_cats)
    proxy_scores, true_scores = [], []

    for _ in range(n_random_subsets):
        subset = list(np.random.choice(n_sources, subset_size, replace=False))

        proxy = estimated_gain_text(g_T, cat_grads, subset)
        true_f1 = get_subset_score(subset, source_sources, target_sources, source_cats)

        proxy_scores.append(proxy)
        true_scores.append(true_f1)
        print(f"  subset={subset}  proxy={proxy:.3f}  true_f1={true_f1:.3f}")

    corr, pval = spearmanr(proxy_scores, true_scores)
    print(f"\nSpearman r={corr:.3f}, p={pval:.3f}")

    if corr < 0.3:
        print("weakly correlated")
    else:
        print("looks useful")

    return corr, proxy_scores, true_scores

def estimated_gain_text(g_T, cat_grads, subset, normalize=True):
    cats = list(cat_grads.keys())
    cat_subset = [cats[i] for i in subset]
    if len(subset) == 0:
        g_subset = torch.zeros_like(g_T)
    else:
        g_subset = sum(cat_grads[loc] for loc in cat_subset)

    if normalize:
        g_subset = F.normalize(g_subset, dim=0) if g_subset.norm() > 0 else g_subset

    gain_subset = F.cosine_similarity(g_T, g_subset, dim=0)

    return (gain_subset).item()

def estimated_marginal_gain_text(g_T, cat_grads, subset, source, normalize=True):
    cats = list(cat_grads.keys())
    cat_subset = [cats[i] for i in subset]

    if len(subset) == 0:
        g_subset = torch.zeros_like(g_T)
    else:
        g_subset = torch.stack([cat_grads[loc] for loc in cat_subset]).mean(0)

    candidate_grad = cat_grads[cats[source]]

    n = len(subset)
    g_subset_plus = (g_subset * n + candidate_grad) / (n + 1)

    if normalize:
        g_T_n = F.normalize(g_T, dim=0)
        g_sub_n = F.normalize(g_subset, dim=0) if g_subset.norm() > 0 else g_subset
        g_plus_n = F.normalize(g_subset_plus, dim=0)
    else:
        g_T_n, g_sub_n, g_plus_n = g_T, g_subset, g_subset_plus

    cos_before = F.cosine_similarity(g_T_n.unsqueeze(0), g_sub_n.unsqueeze(0)).item()
    cos_after = F.cosine_similarity(g_T_n.unsqueeze(0), g_plus_n.unsqueeze(0)).item()
    return cos_after - cos_before

def acs_splice_text(g_T, cat_grads, K=5):
    all_cats = list(cat_grads.keys())
    remaining = all_cats.copy()
    selected = []
    subset_grads = []

    for _ in range(K):
        best_cat, best_gain = None, float('-inf')

        for cat in remaining:
            marginal = estimated_marginal_gain_text(g_T, cat_grads, selected, cat)
            if marginal > best_gain:
                best_gain = marginal
                best_cat = cat

        selected.append(best_cat)
        subset_grads.append(cat_grads[best_cat])
        remaining.remove(best_cat)
        print(f"Added {best_cat}: +{best_gain:.4f}")

    return selected
