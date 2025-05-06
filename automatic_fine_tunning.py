import optuna
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from transformer_pairwise_hinge_loss import (
    load_enriched_user_data, split_users, create_data_splits,
    load_song_metadata, build_vocab, cache_encoded_metadata,
    RecommenderDataset, collate_fn, RecommendationModel,
    evaluate
)

def objective(trial):
    user_emb_dim = trial.suggest_categorical("user_emb_dim", [32, 48, 64, 128])
    text_emb_dim = trial.suggest_categorical("text_emb_dim", [128, 256, 512])
    dropout_p = trial.suggest_float("dropout_p", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    user_records = load_enriched_user_data("best100")
    train_users, dev_users, test_users = split_users(user_records)
    train_examples, dev_eval, _ = create_data_splits(user_records, train_users, dev_users, test_users)

    song_meta_dict = load_song_metadata("songdata")
    vocab = build_vocab(song_meta_dict, min_freq=1)
    max_len = 200
    song_meta_cache = cache_encoded_metadata(song_meta_dict, vocab, max_len)
    user2idx = {uid: i for i, uid in enumerate(sorted({ex[0] for ex in train_examples}))}

    train_dataset = RecommenderDataset(train_examples, user2idx, song_meta_cache, vocab, max_len)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecommendationModel(
        num_users=len(user2idx), user_emb_dim=user_emb_dim,
        vocab_size=len(vocab), text_emb_dim=text_emb_dim,
        num_filters=0, filter_size=0,
        max_len=max_len, dropout_p=dropout_p
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    aux_loss_weight   = 0.4
    ranking_margin    = 1.0
    high_thresh       = 1.0
    high_mse_weight   = 0.1
    aux_criterion     = torch.nn.MSELoss()

    for epoch in range(1, 6):
        model.train()
        for batch in train_loader:
            user_idxs, token_ids, level_nums, judgements, scores = batch
            user_idxs   = user_idxs.to(device)
            token_ids   = token_ids.to(device)
            level_nums  = level_nums.to(device)
            judgements  = judgements.to(device)
            scores      = scores.to(device)

            optimizer.zero_grad()
            score_pred, aux_pred = model(user_idxs, token_ids, level_nums)
            y      = scores
            s_hat  = score_pred
            pairs  = (y.unsqueeze(1) > y.unsqueeze(0)).float()
            diffs  = s_hat.unsqueeze(1) - s_hat.unsqueeze(0)
            hinge  = torch.clamp(ranking_margin - diffs, min=0.0)
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

    dev_acc = evaluate(
        model, dev_eval, user2idx,
        song_meta_cache, vocab, max_len, device
    )

    return dev_acc

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("Best hyperparameters:", study.best_params)
    print("Best dev accuracy:", study.best_value)
