import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms as T
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import f1_score
import config
import os

def wilds_setup():
    data_root = Path("iwildcam-2020-fgvc7")
    with open(data_root / "iwildcam2020_train_annotations.json", "r") as f:
        meta = json.load(f)
    with open(data_root / "iwildcam2020_megadetector_results.json", "r") as f:
        md_meta = json.load(f)

    images = pd.DataFrame(meta["images"])
    annots = pd.DataFrame(meta["annotations"])
    md_df = pd.DataFrame(md_meta["images"])

    img_df = images.merge(annots[["image_id", "category_id"]], left_on="id", right_on="image_id")
    img_df = img_df.merge(md_df, on="id")
    img_df["image_path"] = img_df["file_name"].apply(lambda x: str(Path("train") / x))


    min_images_per_loc = 2000
    loc_counts = img_df["location"].value_counts()
    big_locs = loc_counts[loc_counts >= min_images_per_loc].index
    img_big = img_df[img_df["location"].isin(big_locs)].copy()


    all_locs = img_big["location"].unique()
    rng = np.random.default_rng(42)
    target_locs = rng.choice(all_locs, size=3, replace=False)

    source_df = img_big[~img_big["location"].isin(target_locs)].copy()
    target_full_df = img_big[img_big["location"].isin(target_locs)].copy()


    target_full_df = target_full_df.sample(frac=1.0, random_state=42)
    n_val = int(len(target_full_df) * 0.3)
    target_val_df = target_full_df.iloc[:n_val].copy()
    target_test_df = target_full_df.iloc[n_val:].copy()


    all_labels = sorted(img_big["category_id"].unique())
    label2idx = {lbl: i for i, lbl in enumerate(all_labels)}
    for df in [source_df, target_val_df, target_test_df]:
        df["label"] = df["category_id"].map(label2idx)

    return len(label2idx), data_root, source_df, target_val_df, target_test_df, label2idx


@torch.no_grad()
def precompute_location_features(df, data_root, backbone, device, save_name=None):
    if save_name and os.path.exists(save_name):
        data = torch.load(save_name, weights_only=False)
        return data['feats'], data['labels'], data['locs']

    backbone.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    feats, labels, locs = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Features ({save_name})"):
        try:
            img = Image.open(data_root / row["image_path"]).convert("RGB")

            img_t = transform(img).unsqueeze(0).to(device)

            feat = backbone(img_t).cpu()
            feats.append(feat)
            labels.append(row["label"])
            locs.append(row["location"])
        except Exception as e:
            continue

    out_feats = torch.cat(feats)
    out_labels = torch.tensor(labels)
    out_locs = np.array(locs)

    if save_name:
        torch.save({'feats': out_feats, 'labels': out_labels, 'locs': out_locs}, save_name)

    return out_feats, out_labels, out_locs

def get_balanced_subset(feats, labels, locs, target_locs, n_per_loc=500):
    torch.manual_seed(42)
    all_indices = []

    for loc in target_locs:
        loc_indices = torch.where(torch.tensor(locs == loc))[0]

        perm = torch.randperm(len(loc_indices))
        selected_indices = loc_indices[perm[:n_per_loc]]

        all_indices.append(selected_indices)

    final_indices = torch.cat(all_indices)

    return feats[final_indices], labels[final_indices]

def train_on_subset(loc_indices, source_feats, source_labels, source_locs, num_classes, device, epochs=15):
    torch.manual_seed(42)
    unique_locs = sorted(np.unique(source_locs))
    current_sources = [unique_locs[i] for i in loc_indices]

    X_cpu, y_cpu = get_balanced_subset(
        source_feats,
        source_labels,
        source_locs,
        current_sources,
        n_per_loc=500
    )
    X = X_cpu.to(device)
    y = y_cpu.to(device)

    head = nn.Linear(X.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)

    head.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            idx = perm[i:i+64]
            loss = F.cross_entropy(head(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return head


def evaluate_on_target(head, test_feats, test_labels, device):
    head.eval()
    with torch.no_grad():
        logits = head(test_feats.to(device))
        preds = logits.argmax(1).cpu().numpy()


        targets = test_labels.cpu().numpy()

        macro_f1 = f1_score(targets, preds, average='macro')

        acc = (logits.argmax(1) == test_labels.to(device)).float().mean().item()

    return macro_f1

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
num_classes, data_root, source_df, val_df, test_df, _ = wilds_setup()

config.NUM_CLASSES = num_classes
print("Num Classes Wilds: ", num_classes)
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = nn.Identity()
resnet.to(device)


s_feats, s_labels, s_locs = precompute_location_features(source_df, data_root, resnet, device, "source.pt")
v_feats, v_labels, _ = precompute_location_features(val_df, data_root, resnet, device, "target_val.pt")
t_feats, t_labels, _ = precompute_location_features(test_df, data_root, resnet, device, "target_test.pt")

def get_subset_acc(subset, ctx):
    torch.manual_seed(42)
    np.random.seed(42)
    head = train_on_subset(subset, s_feats, s_labels, s_locs, num_classes, device)
    val_macro_f1 = evaluate_on_target(head, v_feats, v_labels, device)
    test_macro_f1 = evaluate_on_target(head, t_feats, t_labels, device)
    torch.mps.empty_cache()
    return test_macro_f1

def soft_macro_f1_loss(logits, labels, num_classes, eps=1e-7):
    probs = torch.softmax(logits, dim=1)
    y_oh  = F.one_hot(labels, num_classes).float()

    tp = (probs * y_oh).sum(0)
    fp = (probs * (1 - y_oh)).sum(0)
    fn = ((1 - probs) * y_oh).sum(0)

    f1_per_class = (2 * tp) / (2 * tp + fp + fn + eps)
    macro_f1 = f1_per_class.mean()

    return 1.0 - macro_f1

def compute_gradients_f1(feats, labels, head, device, num_classes):
    head.train()
    feats, labels = feats.to(device), labels.to(device)
    head.zero_grad()
    logits = head(feats)
    loss = soft_macro_f1_loss(logits, labels, num_classes)
    loss.backward()
    return torch.cat([p.grad.detach().view(-1) for p in head.parameters()])

def compute_source_gradients_f1(s_feats, s_labels, s_locs,
                                 warm_head, device, num_classes):
    loc_grads = {}
    for loc in tqdm(sorted(np.unique(s_locs)), desc="Source grads (F1)"):
        mask   = s_locs == loc
        feats  = s_feats[mask]
        labels = s_labels[mask]
        if len(feats) == 0:
            continue
        loc_grads[loc] = compute_gradients_f1(
            feats, labels, warm_head, device, num_classes
        )
        loc_grads[loc] = F.normalize(loc_grads[loc], dim=0)
    return loc_grads


def compute_target_gradients_f1(target_feats, target_labels, head, device, num_classes):
    return compute_gradients_f1(target_feats, target_labels, head, device, num_classes)


def precompute_all_gradients(s_feats, s_labels, s_locs,
                              v_feats, v_labels,
                              warm_head, device, num_classes):

    print("Computing source gradients (soft F1)...")
    loc_grads = compute_source_gradients_f1(
        s_feats, s_labels, s_locs, warm_head, device, num_classes
    )

    print("Computing target gradient (soft F1)...")
    target_grad = compute_target_gradients_f1(
        v_feats, v_labels, warm_head, device, num_classes
    )
    target_grad = F.normalize(target_grad, dim=0)

    return loc_grads, target_grad


def estimated_marginal_gain(g_T, loc_grads, g_subset, candidate_loc):
    g_candidate = loc_grads[candidate_loc]
    g_plus      = g_subset + g_candidate


    g_sub_n  = F.normalize(g_subset, dim=0) if g_subset.norm() > 1e-8 \
               else torch.zeros_like(g_T)
    g_plus_n = F.normalize(g_plus, dim=0)

    cos_before = F.cosine_similarity(g_T.unsqueeze(0), g_sub_n.unsqueeze(0)).item()
    cos_after  = F.cosine_similarity(g_T.unsqueeze(0), g_plus_n.unsqueeze(0)).item()

    return cos_after - cos_before, g_plus

def get_grads(epochs=18):
    print("Warm-start head...")
    warm_head = train_on_subset(
        list(range(len(np.unique(s_locs)))),
        s_feats, s_labels, s_locs, num_classes, device, epochs=5
    )

    print("\n Precomputing gradients...")
    loc_grads, target_grad = precompute_all_gradients(
        s_feats, s_labels, s_locs,
        v_feats, v_labels,
        warm_head, device, num_classes
    )
    print(f"  Cached {len(loc_grads)} source location grads | "
          f"grad dim={target_grad.shape[0]:,}")

    return loc_grads, target_grad

def subset_grad(target_grad, loc_grads, subset):
    locs = list(loc_grads.keys())

    if len(subset) == 0:
        g_subset = torch.zeros_like(target_grad)
    else:
        g_subset = torch.stack([loc_grads[locs[i]] for i in subset]).sum(0)

    g_sub_n  = F.normalize(g_subset, dim=0) if g_subset.norm() > 1e-8 \
               else torch.zeros_like(target_grad)

    cos_before = F.cosine_similarity(target_grad.unsqueeze(0), g_sub_n.unsqueeze(0)).item()

    return cos_before
def marginal_gain_estimation(target_grad, loc_grads, subset, candidate):
    locs = list(loc_grads.keys())

    if len(subset) == 0:
        g_subset = torch.zeros_like(target_grad)
    else:
        g_subset = torch.stack([loc_grads[locs[i]] for i in subset]).sum(0)

    g_plus = g_subset + loc_grads[locs[candidate]]

    g_sub_n  = F.normalize(g_subset, dim=0) if g_subset.norm() > 1e-8 \
               else torch.zeros_like(target_grad)
    g_plus_n = F.normalize(g_plus, dim=0)

    cos_before = F.cosine_similarity(target_grad.unsqueeze(0), g_sub_n.unsqueeze(0)).item()
    cos_after  = F.cosine_similarity(target_grad.unsqueeze(0), g_plus_n.unsqueeze(0)).item()

    return cos_after - cos_before
