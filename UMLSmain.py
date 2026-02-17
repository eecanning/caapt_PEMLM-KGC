# UMLSmain.py  (updated)
import argparse
import json
import math
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F

# from your shortlist helper (adjust path if necessary)
from Utils.shortlist_eval import TARGET_RELATION, ALLOWED_TAIL_URIS, filter_triples_for_relation, build_allowed_tail_ids

from tokenizers import Tokenizer
from torch.nn import CrossEntropyLoss
from transformers import BertModel
from Utils.evaluation import *
from torch.utils.data import DataLoader
import os
from Utils.head2tailDataset import *
from Utils.utils import *
from Utils.model import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


@torch.no_grad()
def compute_val_target_ce(model, bert, tokenizer, device, entity_path_or_dict=None, val_triples=None,
                          batch_size=128, *, entity_path=None, **kwargs):
    """
    Compute average cross-entropy over validation triples restricted to the TARGET_RELATION shortlist.

    This function accepts either:
      - positional arg `entity_path_or_dict` which may be either a path-like (str/Path)
        pointing to entities.txt *or* an already-loaded dict mapping entity_uri->id
      - OR the keyword `entity_path=` (kept for backwards compatibility)

    Parameters
    ----------
    model, bert, tokenizer, device : as in training/eval
    entity_path_or_dict / entity_path : path-like or dict (entity_uri -> id)
    val_triples : list of triples (h, r, t) to evaluate (caller should pass list already filtered to TARGET_RELATION)
    batch_size : int
    **kwargs : ignored (kept for forwards compatibility)

    Returns
    -------
    (avg_ce: float, n_examples: int)
    """
    # Accept legacy keyword or positional form
    if entity_path is not None and entity_path_or_dict is None:
        entity_path_or_dict = entity_path

    # Validate val_triples
    if val_triples is None:
        raise RuntimeError("compute_val_target_ce: val_triples must be provided (list of triples).")

    # Load/validate entity2id
    if isinstance(entity_path_or_dict, dict):
        entity2id = entity_path_or_dict
    else:
        try:
            entity2id = read_entity(entity_path_or_dict)
        except Exception as _e:
            raise RuntimeError(f"compute_val_target_ce: failed to read entity mapping from {entity_path_or_dict!r}: {_e}")

    # Build shortlist global ids (uses helper from Utils.shortlist_eval)
    allowed_ids = build_allowed_tail_ids(entity2id)
    if not allowed_ids:
        raise RuntimeError("Allowed-tail shortlist is empty (entity mapping mismatch).")

    allowed_tensor = torch.tensor(allowed_ids, dtype=torch.long, device=device)  # [m]

    # Build inputs/labels in the same 'tail-batch' format used elsewhere
    input_ids_list, labels_list = get_data_from_rawdata(tokenizer, val_triples, entity2id, 'tail-batch')
    if len(input_ids_list) == 0:
        return float("nan"), 0

    total_nll = 0.0
    total_examples = 0

    # batch the inputs
    for start in range(0, len(input_ids_list), batch_size):
        end = min(start + batch_size, len(input_ids_list))
        batch_inputs = input_ids_list[start:end]
        batch_labels = labels_list[start:end]

        inp = torch.tensor(batch_inputs, dtype=torch.long, device=device)  # [B, seq_len]
        with torch.no_grad():
            # prepare embeddings (consistent with training)
            word_embeds = bert.embeddings.word_embeddings(inp)
            pos_embeds = bert.embeddings.position_embeddings(
                torch.arange(0, inp.shape[1], dtype=torch.long).to(device)
            )
            inputs_embeds = word_embeds + pos_embeds

            logits = model(inputs_embeds=inputs_embeds, input_ids=inp)  # [B, label_num]
            # pick shortlist logits
            shortlist_logits = logits.index_select(dim=1, index=allowed_tensor)  # [B, m]
            logp = F.log_softmax(shortlist_logits, dim=1)  # [B, m]

        # map global-label ids -> shortlist positions
        id_to_pos = {int(gid): idx for idx, gid in enumerate(allowed_ids)}
        pos_indices = []
        for g in batch_labels:
            gg = int(g)
            if gg not in id_to_pos:
                # ground-truth NOT in shortlist -> this is unexpected; raise for safety
                raise RuntimeError(f"Ground-truth label {gg} not in ALLOWED_TAIL_URIS shortlist")
            pos_indices.append(id_to_pos[gg])
        pos_tensor = torch.tensor(pos_indices, dtype=torch.long, device=device)  # [B]

        selected_logp = logp[torch.arange(logp.size(0), device=device), pos_tensor]  # [B]
        nll = -selected_logp.sum().item()

        total_nll += nll
        total_examples += logp.size(0)

    avg_ce = total_nll / max(1, total_examples)
    return float(avg_ce), int(total_examples)


def train_val_epochs(bert, model, train_dataloader, val_dataloader, epochs, lr, label_num, device):
    """
    Training loop with validation and TARGET-CE-based early stopping.
    Monitors only TARGET_RELATION cross-entropy (computed by compute_val_target_ce).
    """

    import math
    # keep backwards-compatible locals from arguments
    lr = getattr(arguments, "lr", lr)
    grad_accum_steps = max(1, int(getattr(arguments, "gradient_accumulation_steps", 1)))
    clip_grad_norm = float(getattr(arguments, "clip_grad_norm", 0.0))
    use_fp16 = bool(getattr(arguments, "fp16", False))
    weight_decay = float(getattr(arguments, "weight_decay", 0.0))
    eval_every_epoch_flag = bool(getattr(arguments, "eval_every_epoch", True))
    save_best_model_flag = bool(getattr(arguments, "save_best_model", True))

    # Early stopping config (TARGET_CE-based)
    es_patience = int(getattr(arguments, "early_stopping_patience", 0))         # 0 = disabled
    es_relative_delta = float(getattr(arguments, "es_relative_delta", 0.001))   # require 0.1% improvement by default

    # --- METADATA HANDLING ADDED ---
    # metadata_path is derived from the weight path (same basename, suffix _metadata.json).
    weight_path = getattr(arguments, "weight_path", "parameter/UMLS_PEMLM.pth")
    metadata_path = os.path.splitext(weight_path)[0] + "_metadata.json"
    # initialize metadata structure (will be written out as things happen)
    run_metadata = {
        "weight_path": weight_path,
        "best_epoch": None,
        "stopped_epoch": None,
        "best_val_target_ce": None
    }
    # --- end METADATA HANDLING ---

    # optimizer (prefer AdamW)
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    except Exception:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # fp16 scaler
    scaler = torch.cuda.amp.GradScaler() if use_fp16 else None

    # scheduler (optional)
    scheduler = None
    lr_scheduler_choice = getattr(arguments, "lr_scheduler", "none").lower()
    if lr_scheduler_choice == "linear":
        try:
            from transformers import get_linear_schedule_with_warmup
            steps_per_epoch = math.ceil(len(train_dataloader) / float(grad_accum_steps))
            total_steps = steps_per_epoch * max(1, epochs)
            warmup_steps = max(0, int(0.06 * total_steps))
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=total_steps)
            print(f"[Scheduler] linear scheduler configured: total_steps={total_steps}, warmup={warmup_steps}")
        except Exception as _e:
            print("[Scheduler] failed to configure linear schedule:", _e)
            scheduler = None

    # criterion
    criterion = CrossEntropyLoss()

    # early-stopping runtime state
    # Use es_best numeric sentinel (smaller is better because we monitor CE).
    es_best = float("inf")
    es_wait = 0
    best_epoch = None

    print("[Training] starting epochs:", epochs, "device:", device, "fp16:", use_fp16)

    for epoch in range(epochs):
        model.train()
        total_raw_loss = 0.0
        step_count = 0
        optimizer.zero_grad()

        bar = tqdm(total=len(train_dataloader), desc=f'Epoch{epoch + 1}/{epochs}', ncols=100)

        for iter_idx, batch in enumerate(train_dataloader):
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # prepare embeddings (same as original)
            word_embeds = bert.embeddings.word_embeddings(input_ids)
            pos_embeds = bert.embeddings.position_embeddings(torch.arange(0, input_ids.shape[1],
                                                                           dtype=torch.long).to(device))
            input_embeddings = word_embeds + pos_embeds

            # forward + loss (respect fp16)
            if use_fp16:
                with torch.cuda.amp.autocast():
                    logits = model(inputs_embeds=input_embeddings, input_ids=input_ids)
                    loss = criterion(logits, labels)
            else:
                logits = model(inputs_embeds=input_embeddings, input_ids=input_ids)
                loss = criterion(logits, labels)

            total_raw_loss += float(loss.detach().cpu().item())

            # gradient accumulation
            loss = loss / grad_accum_steps
            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if ((iter_idx + 1) % grad_accum_steps) == 0:
                if use_fp16:
                    scaler.unscale_(optimizer)
                if clip_grad_norm and clip_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                step_count += 1
                if scheduler is not None:
                    scheduler.step()

            bar.set_description(f'train_tail_Epoch {epoch + 1}/{epochs}')
            bar.set_postfix(loss=f'{loss.detach().cpu().item():.6f}')
            bar.update(1)

        bar.close()

        mean_loss = total_raw_loss / max(1, len(train_dataloader))
        print('train_epoch: {}, loss:{}'.format(epoch + 1, mean_loss))
        # write train epoch result
        try:
            with open(arguments.train_result_json_path, 'a') as jsf:
                json.dump({"epoch": epoch + 1, "loss": mean_loss}, jsf); jsf.write('\n')
        except Exception as _e:
            print("[Warning] failed to write train JSON result:", _e)

        # ----------------- validation -----------------
        if eval_every_epoch_flag:
            model.eval()
            total_rank = []
            val_loss = 0.0
            bar = tqdm(total=len(val_dataloader), desc=f'Val {epoch + 1}/{epochs}', ncols=100)
            for iter_idx, batch in enumerate(val_dataloader):
                input_ids, labels = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                word_embeds = bert.embeddings.word_embeddings(input_ids)
                pos_embeds = bert.embeddings.position_embeddings(torch.arange(0, input_ids.shape[1],
                                                                               dtype=torch.long).to(device))
                input_embeddings = word_embeds + pos_embeds

                with torch.no_grad():
                    logits = model(inputs_embeds=input_embeddings, input_ids=input_ids)
                    loss = criterion(logits, labels)

                val_loss += float(loss.detach().cpu().item())

                # evaluation ranks (original behavior)
                r_h = []
                for input_id in input_ids:
                    r = input_id[2].item()
                    h = input_id[1].item()
                    r_token = tokenizer.id_to_token(r)
                    h_token = tokenizer.id_to_token(h)
                    r_h.append((r_token, h_token))

                ranks = evaluation(logits, groundtruth, entity2id, r_h, labels, modes='tail')
                for lbl, rank in zip(labels, ranks):
                    the_rank = torch.where(rank == lbl)[0].item() + 1
                    total_rank.append(the_rank)

                bar.set_description(f'test_tail_Epoch {epoch + 1}/{epochs}')
                bar.set_postfix(loss=f'{loss.detach().cpu().item():.6f}')
                bar.update(1)
            bar.close()

            # compute conventional metrics
            total_rank = torch.tensor(total_rank)
            tail_MR = total_rank.sum().item() / len(total_rank)
            tail_MRR = torch.sum(1 / total_rank).item() / len(total_rank)
            tail_hit10 = torch.sum(total_rank <= 10).item() / len(total_rank)
            tail_hit3 = torch.sum(total_rank <= 3).item() / len(total_rank)
            tail_hit1 = torch.sum(total_rank <= 1).item() / len(total_rank)
            tail_test_loss = val_loss / max(1, len(val_dataloader))

            print('MR:{} ,MRR:{} ,HIT10:{} ,HIT3:{} ,HIT1:{}'.format(
                tail_MR, tail_MRR, tail_hit10, tail_hit3, tail_hit1))

            # write validation metrics json
            try:
                with open(arguments.valid_result_json_path, 'a') as jsf:
                    json.dump({
                        "epoch": epoch + 1,
                        "loss": tail_test_loss,
                        "MR": tail_MR,
                        "MRR": tail_MRR,
                        "Hit@10": tail_hit10,
                        "Hit@3": tail_hit3,
                        "Hit@1": tail_hit1
                    }, jsf); jsf.write('\n')
            except Exception as _e:
                print("[Warning] failed to write valid JSON result:", _e)

            # ----------------- TARGET-RELATION CE (for early stopping) -----------------
            val_ce = float("nan")
            n_val_ce = 0
            try:
                # read validation triples and filter to TARGET_RELATION
                val_triples_all = read_triplets_from_txt(arguments.valid_data_path)
                val_target_triples = filter_triples_for_relation(val_triples_all, TARGET_RELATION)
                if len(val_target_triples) > 0:
                    val_ce, n_val_ce = compute_val_target_ce(
                        model=model, bert=bert, tokenizer=tokenizer,
                        device=device, entity_path=arguments.entity_path,
                        val_triples=val_target_triples,
                        batch_size=int(getattr(arguments, "val_batch_size", 128))
                    )
                    print(f"[TargetCE] val_target_CE={val_ce:.6f}  (n={n_val_ce})")
                else:
                    print("[TargetCE] no validation triples for TARGET_RELATION found; skipping TARGET CE computation.")
            except Exception as _e:
                val_ce = float("nan"); n_val_ce = 0
                print("[TargetCE] error computing target CE:", type(_e).__name__, _e)

            # decide improvement: smaller CE is better; require finite numbers
            improved = False
            if math.isfinite(val_ce):
                # first time if es_best is +inf this will be True (unless delta is > 0)
                threshold = es_best * (1.0 - float(es_relative_delta))
                improved = (val_ce < threshold)
            else:
                improved = False

            if improved:
                es_best = float(val_ce)
                es_wait = 0
                best_epoch = epoch
                if save_best_model_flag:
                    try:
                        torch.save(model.state_dict(), arguments.weight_path)
                        print("model saved (improved TARGET_CE -> {:.6f})".format(val_ce))
                        # --- METADATA: update best info and write JSON ---
                        run_metadata["best_epoch"] = int(best_epoch) + 1   # human-friendly 1-based epoch number
                        run_metadata["best_val_target_ce"] = float(es_best)
                        # do not yet set stopped_epoch here (we don't know if/when training will stop)
                        try:
                            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
                            with open(metadata_path, "w", encoding="utf-8") as mf:
                                json.dump(run_metadata, mf, indent=2)
                        except Exception as _mf_e:
                            print("[Metadata] failed to write metadata JSON:", _mf_e)
                        # --- end metadata write ---
                    except Exception as _e:
                        print("[EarlyStopping] failed to save model:", type(_e).__name__, _e)
            else:
                es_wait += 1
                print(f"[EarlyStopping] TARGET_CE did not improve (wait={es_wait}/{es_patience})")
                if es_patience and es_wait >= es_patience:
                    print(f"[EarlyStopping] patience exceeded ({es_patience}) — stopping training early.")
                    # --- METADATA: record stopped epoch and write final metadata ---
                    run_metadata["stopped_epoch"] = int(epoch) + 1  # 1-based epoch index when stopping
                    # ensure best_epoch has a 1-based value if present
                    if run_metadata["best_epoch"] is None and best_epoch is not None:
                        run_metadata["best_epoch"] = int(best_epoch) + 1
                    try:
                        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
                        with open(metadata_path, "w", encoding="utf-8") as mf:
                            json.dump(run_metadata, mf, indent=2)
                        print(f"[Metadata] written final metadata to {metadata_path}")
                    except Exception as _mf_e:
                        print("[Metadata] failed to write metadata JSON on early stop:", _mf_e)
                    # --- end metadata write ---
                    return
        else:
            # no validation this epoch
            print(f"[Info] Skipping validation for epoch {epoch + 1} (eval_every_epoch=False)")

    # If we exit the loop normally (no early stop), write final metadata
    if run_metadata.get("stopped_epoch") is None:
        try:
            run_metadata["stopped_epoch"] = int(epochs)
            if run_metadata.get("best_epoch") is None and best_epoch is not None:
                run_metadata["best_epoch"] = int(best_epoch) + 1
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            with open(metadata_path, "w", encoding="utf-8") as mf:
                json.dump(run_metadata, mf, indent=2)
            print(f"[Metadata] final metadata written to {metadata_path}")
        except Exception as _mf_e:
            print("[Metadata] failed to write final metadata JSON:", _mf_e)

    # end epochs
    print("[Training] finished all epochs or early-stopped.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="UMLS training and validation")
    parser.add_argument('--train_batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=64, help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for training')
    parser.add_argument('--tran_lr', type=float, default=1e-3, help='Learning rate for transModel')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--weight_path', type=str, default='parameter/UMLS_PEMLM.pth', help='model_weight_path')
    parser.add_argument('--model_path', type=str, default='bert-base-uncased', help='original model path')
    parser.add_argument('--embedding_path', type=str, default='model/UMLS_word_embeddings.pt', help='UMLS embedding path')
    parser.add_argument('--tokenizer_path', type=str, default='model/UMLS_tokenizer.json', help='tokenizer path')
    parser.add_argument('--entity_path', type=str, default='data/UMLS/entities.txt', help='entity path')
    parser.add_argument('--relation_path', type=str, default='data/UMLS/reverse_relations.txt', help='entity path')
    parser.add_argument('--entity2text_path', type=str, default='data/UMLS/entity2textlong.txt', help='entity2text path')
    parser.add_argument('--relation2text_path', type=str, default='data/UMLS/reverse_relation2text.txt', help='relation2text path')
    parser.add_argument('--train_data_path', type=str, default='data/UMLS/train.tsv', help='train data path')
    parser.add_argument('--valid_data_path', type=str, default='data/UMLS/dev.tsv', help='valid data path')
    parser.add_argument('--test_data_path', type=str, default='data/UMLS/test.tsv', help='test data path')
    parser.add_argument('--hidden_size', type=int, default=768, help='hidden size')
    parser.add_argument('--num_attention_heads', type=int, default=4, help='num_attention_heads')
    parser.add_argument('--num_hidden_layers', type=int, default=12, help='num_hidden_layers')
    parser.add_argument('--max_length', type=int, default=128, help='max_length')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--train_result_json_path', type=str, default='log/UMLS/UMLS_PEMLM_trainResult.json', help='train_result_json_path')
    parser.add_argument('--valid_result_json_path', type=str, default='log/UMLS/UMLS_PEMLM_valResult.json', help='valid_result_json_path')
    parser.add_argument('--test_result_json_path', type=str, default='log/UMLS/UMLS_PEMLM_testResult.json', help='test_result_json_path')

    # early stopping args
    parser.add_argument('--early_stopping_patience', type=int, default=0, help='Patience for early stopping (0 = disabled)')
    parser.add_argument('--early_stopping_mode', type=str, default='max', help="'max' or 'min' for improvement direction")
    parser.add_argument('--early_stopping_metric', type=str, default='MRR', help='Metric name to monitor for early stopping (default: MRR)')

    # additional training/optimizer flags requested
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Accumulate gradients this many steps before optimizer.step()')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (L2) for optimizer')
    parser.add_argument('--lr_scheduler', type=str, default='none',
                        help="Learning rate scheduler: 'none' or 'linear'")
    parser.add_argument('--clip_grad_norm', type=float, default=0.0,
                        help='Max norm for gradient clipping (0 = disabled)')
    parser.add_argument('--save_best_model', type=lambda x: x.lower() == 'true', default=True,
                        help='Whether to save best model when validation improves (True/False)')
    parser.add_argument('--eval_every_epoch', type=lambda x: x.lower() == 'true', default=True,
                        help='Run validation every epoch (True/False)')
    parser.add_argument('--fp16', type=lambda x: x.lower() == 'true', default=False,
                        help='Use mixed precision (torch.cuda.amp) if True')
    parser.add_argument('--es_relative_delta', type=float, default=0.001,
                        help='Relative improvement threshold for early stopping (e.g. 0.001 = 0.1%)')

    arguments = parser.parse_args()
    epochs = arguments.epochs
    lr = arguments.lr
    device = arguments.device

    # load data and prepare datasets
    train_data = read_triplets_from_txt(arguments.train_data_path)
    valid_data = read_triplets_from_txt(arguments.valid_data_path)
    test_data = read_triplets_from_txt(arguments.test_data_path)

    # NOTE (patched): The input TSVs already include inverse relations (inv_...)
    # so we should NOT generate additional 'be_' reversed triples here —
    # generating them would create relation tokens like 'be_inv_...' not present
    # in the tokenizer/embeddings and would cause None token ids.
    # Use the provided train triples as-is (they already contain the inverses).
    onlyTail_train_data = train_data
    onlyTail_val_data = valid_data
    onlyTail_test_data = test_data

    entity2id = read_entity(arguments.entity_path)
    entity_set = set(entity2id.keys())
    label_num = len(entity2id)
    groundtruth = count_groundtruth(onlyTail_train_data, onlyTail_val_data, onlyTail_test_data)

    Bert_model = BertModel.from_pretrained(arguments.model_path)

    tokenizer = Tokenizer.from_file(arguments.tokenizer_path)
    vocab = tokenizer.get_vocab()
    entity2id = read_entity(arguments.entity_path)
    label_num = len(entity2id)
    hidden_size = arguments.hidden_size
    new_word_embeddings_weight = torch.load(arguments.embedding_path)
    position_embeddings_weight = Bert_model.embeddings.position_embeddings.weight

    # if weight exists, load it
    weight_path = arguments.weight_path
    if os.path.exists(weight_path):
        Bert_model.embeddings.word_embeddings.weight = torch.nn.Parameter(new_word_embeddings_weight)
        model = MainModel(Bert_model, hidden_size, label_num, tokenizer, device)
        model.load_state_dict(torch.load(weight_path))
        print('model has loaded')
    else:
        print('model dont exists')
        Bert_model.embeddings.word_embeddings.weight = torch.nn.Parameter(new_word_embeddings_weight)
        model = MainModel(Bert_model, hidden_size, label_num, tokenizer, device)
        print('init model.')

    # train / test datasets
    input_ids, labels = get_data_from_rawdata(tokenizer, onlyTail_train_data, entity2id, 'tail-batch')
    train_dataset = OnlyTailDataset(input_ids, labels)

    input_ids, labels = get_data_from_rawdata(tokenizer, onlyTail_test_data, entity2id, 'tail-batch')
    test_dataset = OnlyTailDataset(input_ids, labels)

    train_dataloader = DataLoader(train_dataset, batch_size=arguments.train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=arguments.val_batch_size, shuffle=False)

    # Run training + validation
    train_val_epochs(Bert_model, model, train_dataloader, test_dataloader, epochs, lr, label_num, device)
