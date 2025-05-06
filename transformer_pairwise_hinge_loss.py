import os
import json
import random
import re
import numpy as np
from collections import defaultdict
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

TOKEN_REGEX = re.compile(r'\w+|[^\w\s]', re.UNICODE)

def tokenize(text):
    return TOKEN_REGEX.findall(text)

def build_vocab(song_meta_dict, min_freq=1):
    freq = {}
    for text in song_meta_dict.values():
        for token in tokenize(text):
            freq[token] = freq.get(token, 0) + 1
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, count in freq.items():
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab

def encode_text(text, vocab, max_len):
    tokens = tokenize(text)
    indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if len(indices) < max_len:
        indices += [vocab["<PAD>"]] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices

def filter_metadata(text):
    ignore_prefixes = ["&title=", "&artist=", "&wholebpm=", "&demo_seek=", "&demo_len=", "&lv_7=", "&inote_7="]
    lines = text.splitlines()
    filtered = [line for line in lines if not any(line.startswith(prefix) for prefix in ignore_prefixes)]
    return "\n".join(filtered)

def cache_encoded_metadata(song_meta_dict, vocab, max_len):
    cached = {}
    for title, text in song_meta_dict.items():
        cached[title] = encode_text(text, vocab, max_len)
    return cached

def load_enriched_user_data(user_data_dir):
    user_records = {}
    for fname in os.listdir(user_data_dir):
        if fname.endswith('.json'):
            filepath = os.path.join(user_data_dir, fname)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue

            body = data.get("body", {})
            pbs = body.get("pbs", [])
            charts = body.get("charts", [])
            songs = body.get("songs", [])
            if not pbs or not charts or not songs:
                continue

            charts_map = {rec["chartID"]: rec for rec in charts}
            songs_map = {str(rec.get("id", "")): rec for rec in songs}

            user_id = pbs[0].get("userID")
            if user_id is None:
                continue
            records = []
            for rec in pbs:
                rating = rec.get("scoreData", {}).get("percent", 0) / 100.0
                judgements = rec.get("scoreData", {}).get("judgements", {})
                j_vector = [
                    float(judgements.get("pcrit", 0)),
                    float(judgements.get("perfect", 0)),
                    float(judgements.get("great", 0)),
                    float(judgements.get("good", 0)),
                    float(judgements.get("miss", 0))
                ]
                chartID = rec.get("chartID")
                chart_rec = charts_map.get(chartID, {})
                levelNum = float(chart_rec.get("levelNum", 0))
                songID = str(rec.get("songID", ""))
                song_rec = songs_map.get(songID, {})
                song_title = song_rec.get("title", songID)
                score = levelNum * rating
                records.append((user_id, song_title, levelNum, j_vector, rating, score))
            records = sorted(records, key=lambda x: x[5], reverse=True)
            user_records[user_id] = records
    return user_records

def load_song_metadata(song_data_dir):
    song_meta = {}
    for root, dirs, files in os.walk(song_data_dir):
        if "maidata.txt" in files:
            song_title = os.path.basename(root)
            filepath = os.path.join(root, "maidata.txt")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                filtered_text = filter_metadata(text)
                song_meta[song_title] = filtered_text
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
    return song_meta

def split_users(user_records):
    all_users = list(user_records.keys())
    target_user = 2243
    if target_user in all_users:
        all_users.remove(target_user)
    random.shuffle(all_users)
    train_users = set(all_users[:200])
    dev_users = set(all_users[200:300])
    test_users = set(all_users[300:])
    if target_user in user_records:
        test_users.add(target_user)
    return train_users, dev_users, test_users

def create_data_splits(user_records, train_users, dev_users, test_users):
    train_examples = []
    dev_eval = {}
    test_eval = {}
    for user, records in user_records.items():
        if user in train_users:
            train_examples.extend(records)
        else:
            ground_truth = records[:20] if len(records) >= 20 else records[:]
            training_recs = records[20:] if len(records) >= 20 else []
            train_examples.extend(training_recs)
            eval_candidates = list({(r[1], r[2]) for r in (ground_truth + training_recs)})
            gt_candidates = [(r[1], r[2], r[5]) for r in ground_truth]
            if user in dev_users:
                dev_eval[user] = {"ground_truth": gt_candidates, "candidates": eval_candidates}
            elif user in test_users:
                test_eval[user] = {"ground_truth": gt_candidates, "candidates": eval_candidates}
    return train_examples, dev_eval, test_eval

class RecommenderDataset(Dataset):
    def __init__(self, examples, user2idx, song_meta_cache, vocab, max_len):
        self.examples = examples
        self.user2idx = user2idx
        self.song_meta_cache = song_meta_cache
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        user_id, song_title, levelNum, judgements, rating, score = self.examples[idx]
        user_idx = self.user2idx[user_id]
        encoded = self.song_meta_cache.get(song_title)
        if encoded is None:
            encoded = encode_text("", self.vocab, self.max_len)
        token_ids = torch.tensor(encoded, dtype=torch.long)
        level_num = torch.tensor(levelNum, dtype=torch.float)
        judgements_tensor = torch.tensor(judgements, dtype=torch.float)
        score = torch.tensor(score, dtype=torch.float)
        return user_idx, token_ids, level_num, judgements_tensor, score

def collate_fn(batch):
    user_idxs = torch.tensor([item[0] for item in batch], dtype=torch.long)
    token_ids = torch.stack([item[1] for item in batch])
    level_nums = torch.tensor([item[2] for item in batch], dtype=torch.float)
    judgements = torch.stack([item[3] for item in batch])
    scores = torch.tensor([item[4] for item in batch], dtype=torch.float)
    return user_idxs, token_ids, level_nums, judgements, scores

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class SongEncoderTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, max_len, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=0.1, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, output_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        out = self.fc(x)
        return out

class RecommendationModel(nn.Module):
    def __init__(self, num_users, user_emb_dim, vocab_size, text_emb_dim, num_filters, filter_size, max_len, dropout_p=0.5, score_scale=1.02):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, user_emb_dim)
        self.song_encoder = SongEncoderTransformer(vocab_size, embed_dim=text_emb_dim, nhead=2, num_layers=2, max_len=max_len, output_dim=user_emb_dim)
        self.numeric_encoder = nn.Linear(1, user_emb_dim)
        self.attention = nn.MultiheadAttention(embed_dim=user_emb_dim, num_heads=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(user_emb_dim * 3, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, 1)
        )
        self.aux_fc = nn.Sequential(
            nn.Linear(user_emb_dim * 3, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(32, 5)
        )
        self.score_scale = score_scale
    
    def forward(self, user_idx, song_tokens, numeric_feature):
        user_emb = self.user_embedding(user_idx)
        song_emb = self.song_encoder(song_tokens)
        query = user_emb.unsqueeze(1)
        key = song_emb.unsqueeze(1)
        value = song_emb.unsqueeze(1)
        attn_output, _ = self.attention(query, key, value)
        attn_output = attn_output.squeeze(1)
        combined_song = song_emb + attn_output
        numeric_emb = self.numeric_encoder(numeric_feature.unsqueeze(1))
        combined = torch.cat([user_emb, combined_song, numeric_emb], dim=1)
        score = self.fc(combined).squeeze(1)
        score = score * self.score_scale
        aux = self.aux_fc(combined)
        return score, aux


def evaluate(model, eval_data, user2idx, song_meta_cache, vocab, max_len, device):
    model.eval()
    accuracies = []
    with torch.no_grad():
        for user_id, info in eval_data.items():
            if user_id not in user2idx:
                continue
            user_idx = torch.tensor([user2idx[user_id]], dtype=torch.long).to(device)
            candidate_list = info["candidates"]
            token_batch = []
            numeric_features = []
            for title, lvl in candidate_list:
                encoded = song_meta_cache.get(title)
                if encoded is None:
                    encoded = encode_text("", vocab, max_len)
                token_batch.append(encoded)
                numeric_features.append(lvl)
            token_batch = torch.tensor(token_batch, dtype=torch.long).to(device)
            numeric_features = torch.tensor(numeric_features, dtype=torch.float).to(device)
            preds, _ = model(user_idx.repeat(token_batch.size(0)), token_batch, numeric_features)
            preds = preds.cpu().numpy().flatten()
            top_n = min(20, len(candidate_list))
            top20_idx = np.argsort(preds)[-top_n:][::-1]
            predicted_top20 = [candidate_list[i][0] for i in top20_idx]
            ground_truth = [t for t, _, _ in info["ground_truth"]]
            match_count = sum(1 for title in predicted_top20 if title in ground_truth)
            user_accuracy = match_count / 20.0
            accuracies.append(user_accuracy)
    return np.mean(accuracies) if accuracies else 0

def compute_popularity_baseline(train_examples):
    song_scores = {}
    song_counts = {}
    for ex in train_examples:
        song_title = ex[1]
        score = ex[5]
        if song_title in song_scores:
            song_scores[song_title] += score
            song_counts[song_title] += 1
        else:
            song_scores[song_title] = score
            song_counts[song_title] = 1
    baseline_scores = {song: song_scores[song] / song_counts[song] for song in song_scores}
    return baseline_scores

def evaluate_baseline(baseline_scores, eval_data):
    accuracies = []
    for user_id, info in eval_data.items():
        candidate_list = info["candidates"]
        candidate_vals = [baseline_scores.get(title, 0.0) for title, lvl in candidate_list]
        candidate_vals = np.array(candidate_vals)
        top_n = min(20, len(candidate_list))
        top20_idx = np.argsort(candidate_vals)[-top_n:][::-1]
        predicted_top20 = [candidate_list[i][0] for i in top20_idx]
        ground_truth = [t for t, _, _ in info["ground_truth"]]
        match_count = sum(1 for title in predicted_top20 if title in ground_truth)
        accuracies.append(match_count / 20.0)
    return np.mean(accuracies) if accuracies else 0

def get_song_feature(song_title, lvl, model, song_meta_cache, vocab, max_len, device):
    model.song_encoder.eval()
    encoded = song_meta_cache.get(song_title)
    if encoded is None:
        encoded = encode_text("", vocab, max_len)
    token_ids = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        song_emb = model.song_encoder(token_ids)
    song_emb = song_emb.squeeze(0).cpu().numpy()
    numeric_feature = np.array([lvl], dtype=np.float32)
    return np.concatenate([song_emb, numeric_feature])

def evaluate_naive_bayes(user_records, eval_data, model, song_meta_cache, vocab, max_len, device):
    model.eval()
    accuracies = []
    epsilon = 1e-6
    for user_id, info in eval_data.items():
        if user_id not in user_records:
            continue
        records = user_records[user_id]
        train_recs = records[20:]
        if len(train_recs) == 0:
            continue
        features = [get_song_feature(rec[1], rec[2], model, song_meta_cache, vocab, max_len, device)
                    for rec in train_recs]
        features = np.stack(features, axis=0)
        mu = features.mean(axis=0)
        var = features.var(axis=0)
        candidate_list = info["candidates"]
        candidate_scores = []
        for title, lvl in candidate_list:
            feat = get_song_feature(title, lvl, model, song_meta_cache, vocab, max_len, device)
            log_probs = -0.5 * np.log(2 * np.pi * (var + epsilon)) - ((feat - mu) ** 2) / (2 * (var + epsilon))
            candidate_scores.append(log_probs.sum())
        candidate_scores = np.array(candidate_scores)
        top_n = min(20, len(candidate_list))
        top20_idx = np.argsort(candidate_scores)[-top_n:][::-1]
        predicted_top20 = [candidate_list[i][0] for i in top20_idx]
        ground_truth = [t for t, _, _ in info["ground_truth"]]
        match_count = sum(1 for title in predicted_top20 if title in ground_truth)
        accuracies.append(match_count / 20.0)
    return np.mean(accuracies) if accuracies else 0

def evaluate_mf(model, eval_data, user2idx, song2idx, device):
    model.eval()
    accuracies = []
    with torch.no_grad():
        for user_id, info in eval_data.items():
            if user_id not in user2idx:
                continue
            uid = user2idx[user_id]
            candidate_list = info["candidates"]
            preds = []
            valid_candidates = []
            for title, lvl in candidate_list:
                if title in song2idx:
                    song_idx = song2idx[title]
                    user_tensor = torch.tensor([uid], dtype=torch.long, device=device)
                    song_tensor = torch.tensor([song_idx], dtype=torch.long, device=device)
                    pred = model(user_tensor, song_tensor).item()
                    preds.append(pred)
                    valid_candidates.append(title)
                else:
                    preds.append(-9999.0)
                    valid_candidates.append(title)
            preds = torch.tensor(preds, dtype=torch.float32)
            top_n = min(20, len(candidate_list))
            top20_idx = torch.argsort(preds, descending=True)[:top_n]
            predicted_top20 = [valid_candidates[i] for i in top20_idx]
            ground_truth = [t for t, _, _ in info["ground_truth"]]
            match_count = sum(1 for title in predicted_top20 if title in ground_truth)
            accuracies.append(match_count / 20.0)
    return np.mean(accuracies) if accuracies else 0

class MFModel(nn.Module):
    def __init__(self, num_users, num_songs, latent_dim=32):
        super(MFModel, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, latent_dim)
        self.song_embeddings = nn.Embedding(num_songs, latent_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.song_bias = nn.Embedding(num_songs, 1)
        
    def forward(self, user_idxs, song_idxs):
        user_emb = self.user_embeddings(user_idxs)
        song_emb = self.song_embeddings(song_idxs)
        dot = (user_emb * song_emb).sum(dim=1, keepdim=True)
        bias = self.user_bias(user_idxs) + self.song_bias(song_idxs)
        score = dot + bias
        return score.squeeze(1)

class MFDataset(Dataset):
    def __init__(self, examples, user2idx, song2idx):
        self.data = []
        for ex in examples:
            user, song, lvl, judgements, rating, score = ex
            if user in user2idx and song in song2idx:
                self.data.append((user2idx[user], song2idx[song], float(score)))
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        user_idx, song_idx, score = self.data[idx]
        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(song_idx, dtype=torch.long),
            torch.tensor(score, dtype=torch.float32)
        )

def detailed_evaluation_test(model, test_eval, user_records, user2idx, song_meta_cache, vocab, max_len, device, score_mean, score_std, output_path="detailed_test_output.txt"):
    test_users = list(test_eval.keys())
    selected_users = []
    target_user = 2243
    if target_user in test_users or str(target_user) in test_users:
        selected_users.append(str(target_user) if str(target_user) in test_users else target_user)
    remaining = [u for u in test_users if u not in selected_users]
    random.shuffle(remaining)
    selected_users.extend(remaining[:(10 - len(selected_users))])
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("--- Detailed Test Evaluation for Selected Users ---\n")
        model.eval()
        with torch.no_grad():
            for user in selected_users:
                f.write(f"\nUser: {user}\n")
                if user not in user_records:
                    f.write("No user records found.\n")
                    continue
                gt_records = user_records[user][:20]
                f.write("Ground Truth (Top 20):\n")
                for rec in gt_records:
                    f.write(f"  Song: {rec[1]:30s} | Level: {rec[2]:.2f} | Score: {rec[5]:.4f}\n")
                
                eval_info = test_eval[user]
                candidate_list = eval_info["candidates"]
                token_batch = []
                numeric_features = []
                for title, lvl in candidate_list:
                    encoded = song_meta_cache.get(title)
                    if encoded is None:
                        encoded = encode_text("", vocab, max_len)
                    token_batch.append(encoded)
                    numeric_features.append(lvl)
                token_batch = torch.tensor(token_batch, dtype=torch.long).to(device)
                numeric_features = torch.tensor(numeric_features, dtype=torch.float).to(device)
                user_idx = torch.tensor([user2idx[user]], dtype=torch.long).to(device)
                preds, _ = model(user_idx.repeat(token_batch.size(0)), token_batch, numeric_features)
                preds = preds.cpu().numpy().flatten()
                denorm_preds = preds * score_std + score_mean
                top_n = min(20, len(candidate_list))
                top20_idx = np.argsort(preds)[-top_n:][::-1]
                
                gt_set = set(rec[1] for rec in gt_records)
                full_gt_map = {rec[1]: rec[5] for rec in user_records[user]}
                
                f.write("\nPredicted Top 20:\n")
                for idx in top20_idx:
                    song_title, lvl = candidate_list[idx]
                    pred_score = denorm_preds[idx]
                    gt_score = full_gt_map.get(song_title, 0.0)
                    match_label = " (MATCH)" if song_title in gt_set else ""
                    f.write(f"  Song: {song_title:30s} | Level: {lvl:.2f} | Predicted Score: {pred_score:.4f} | GT Score: {gt_score:.4f}{match_label}\n")

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    user_data_dir = "best100"
    song_data_dir = "songdata"
    
    print("Loading enriched user performance data...")
    user_records = load_enriched_user_data(user_data_dir)
    print(f"Loaded enriched data for {len(user_records)} users.")
    
    train_users, dev_users, test_users = split_users(user_records)
    print(f"Training users: {len(train_users)}, Dev users: {len(dev_users)}, Test users: {len(test_users)}")
    
    train_examples, dev_eval, test_eval = create_data_splits(user_records, train_users, dev_users, test_users)
    print(f"Total training examples: {len(train_examples)}")
    
    all_training_users = set(ex[0] for ex in train_examples)
    user2idx = {uid: i for i, uid in enumerate(sorted(all_training_users))}
    
    print("Loading song metadata from maidata.txt files...")
    song_meta_dict = load_song_metadata(song_data_dir)
    print(f"Loaded metadata for {len(song_meta_dict)} songs.")
    
    print("Building vocabulary from song metadata...")
    vocab = build_vocab(song_meta_dict, min_freq=1)
    print(f"Vocabulary size: {len(vocab)}")
    max_len = 200
    
    print("Caching encoded song metadata...")
    song_meta_cache = cache_encoded_metadata(song_meta_dict, vocab, max_len)
    
    song2idx = {song: i for i, song in enumerate(sorted(song_meta_dict.keys()))}
    
    train_scores = [ex[5] for ex in train_examples]
    score_mean = np.mean(train_scores)
    score_std = np.std(train_scores)
    print(f"Score normalization: mean = {score_mean:.4f}, std = {score_std:.4f}")
    train_examples = [(u, s, lvl, judgements, rating, ((lvl * rating) - score_mean) / score_std)
                      for (u, s, lvl, judgements, rating, score) in train_examples]
    
    user_emb_dim = 128
    text_emb_dim = 256
    num_filters = 64
    filter_size = 3
    batch_size = 64
    num_epochs = 5
    learning_rate = 0.0012922274696409307
    aux_loss_weight = 0.4
    dropout_p = 0.1580414621880086
    ranking_margin = 1.0
    high_thresh = 1.0
    high_mse_weight = 0.1
    score_scale = 1.015
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dataset = RecommenderDataset(train_examples, user2idx, song_meta_cache, vocab, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    model = RecommendationModel(len(user2idx), user_emb_dim, len(vocab), text_emb_dim, num_filters, filter_size, max_len, dropout_p, score_scale)
    model = model.to(device)
    
    print("\n--- Evaluating Baselines BEFORE Training Main Model ---")
    baseline_scores = compute_popularity_baseline(train_examples)
    baseline_dev_acc = evaluate_baseline(baseline_scores, dev_eval)
    baseline_test_acc = evaluate_baseline(baseline_scores, test_eval)
    print(f"Popularity Baseline Dev Accuracy = {baseline_dev_acc:.4f}")
    print(f"Popularity Baseline Test Accuracy = {baseline_test_acc:.4f}")
    
    naive_bayes_dev_acc = evaluate_naive_bayes(user_records, dev_eval, model, song_meta_cache, vocab, max_len, device)
    naive_bayes_test_acc = evaluate_naive_bayes(user_records, test_eval, model, song_meta_cache, vocab, max_len, device)
    print(f"Naïve Bayes Baseline Dev Accuracy = {naive_bayes_dev_acc:.4f}")
    print(f"Naïve Bayes Baseline Test Accuracy = {naive_bayes_test_acc:.4f}")
    
    aux_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    best_dev_acc = 0.0
    best_model_state = None
    best_epoch = 0
    
    print("\nStarting training for 5 epochs (main model)...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            user_idxs, token_ids, level_nums, judgements, scores = batch
            user_idxs = user_idxs.to(device)
            token_ids = token_ids.to(device)
            level_nums = level_nums.to(device)
            judgements = judgements.to(device)
            scores = scores.to(device)
            
            optimizer.zero_grad()
            score_pred, aux_pred = model(user_idxs, token_ids, level_nums)
            
            y = scores
            s_hat = score_pred
            pairs = (y.unsqueeze(1) > y.unsqueeze(0)).float()
            diffs = s_hat.unsqueeze(1) - s_hat.unsqueeze(0)
            hinge = torch.clamp(ranking_margin - diffs, min=0.0)
            rank_loss = (hinge * pairs).mean()
            
            high_mask = y > high_thresh
            if high_mask.sum() > 1:
                mse_high = ((s_hat[high_mask] - y[high_mask]).pow(2)).mean()
            else:
                mse_high = torch.tensor(0.0, device=device)
            loss_score = rank_loss + high_mse_weight * mse_high
            
            loss_aux = aux_criterion(aux_pred, judgements)
            loss = loss_score + aux_loss_weight * loss_aux
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item() * user_idxs.size(0)
        avg_loss = total_loss / len(train_dataset)
        dev_acc = evaluate(model, dev_eval, user2idx, song_meta_cache, vocab, max_len, device)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Dev Accuracy = {dev_acc:.4f}")
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
    
    print(f"\nBest dev accuracy (main model): {best_dev_acc:.4f} at epoch {best_epoch}")
    model.load_state_dict(best_model_state)
    test_acc = evaluate(model, test_eval, user2idx, song_meta_cache, vocab, max_len, device)
    print(f"Final Test Accuracy (main model) = {test_acc:.4f}")
    
    detailed_evaluation_test(model, test_eval, user_records, user2idx, song_meta_cache, vocab, max_len, device, score_mean, score_std, output_path="user_details_score_normalized_transformer_hinge.txt")
    
    print("\nTraining Matrix Factorization (MF) Baseline...")
    mf_dataset = MFDataset(train_examples, user2idx, song2idx)
    mf_loader = DataLoader(mf_dataset, batch_size=64, shuffle=True)
    num_users = len(user2idx)
    num_songs = len(song2idx)
    mf_model = MFModel(num_users, num_songs, latent_dim=32).to(device)
    mf_optimizer = optim.Adam(mf_model.parameters(), lr=0.005)
    mf_criterion = nn.MSELoss()
    mf_epochs = 15
    best_mf_dev_acc = 0.0
    best_mf_model_state = None
    best_mf_epoch = 0
    for epoch in range(1, mf_epochs + 1):
        mf_model.train()
        total_loss = 0.0
        for batch in mf_loader:
            u_idxs, s_idxs, scores = batch
            u_idxs = u_idxs.to(device)
            s_idxs = s_idxs.to(device)
            scores = scores.to(device)
            mf_optimizer.zero_grad()
            preds = mf_model(u_idxs, s_idxs)
            loss = mf_criterion(preds, scores)
            loss.backward()
            mf_optimizer.step()
            total_loss += loss.detach().item() * u_idxs.size(0)
        avg_loss = total_loss / len(mf_dataset)
        mf_dev_acc = evaluate_mf(mf_model, dev_eval, user2idx, song2idx, device)
        print(f"MF Epoch {epoch}: Loss = {avg_loss:.4f}, Dev Accuracy = {mf_dev_acc:.4f}")
        if mf_dev_acc > best_mf_dev_acc:
            best_mf_dev_acc = mf_dev_acc
            best_mf_model_state = copy.deepcopy(mf_model.state_dict())
            best_mf_epoch = epoch
    
    print(f"\nBest MF dev accuracy: {best_mf_dev_acc:.4f} at epoch {best_mf_epoch}")
    mf_model.load_state_dict(best_mf_model_state)
    mf_test_acc = evaluate_mf(mf_model, test_eval, user2idx, song2idx, device)
    print(f"MF Baseline Test Accuracy (best model) = {mf_test_acc:.4f}")
    
    print("\nTraining complete.")

if __name__ == "__main__":
    main()

