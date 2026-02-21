# File: PEMLM-KGC/Utils/shortlist_eval.py
import os
import torch
from typing import List, Tuple, Dict

# ---------- User settings (hardcoded shortlist & target relation) ----------
TARGET_RELATION = "https://w3id.org/caapt/decision-making#task_2_tsd"
ALLOWED_TAIL_URIS = [
    "https://w3id.org/caapt/terms#black_uc1",
        "https://w3id.org/caapt/terms#black_uc2",
        "https://w3id.org/caapt/terms#black_uc3",
        "https://w3id.org/caapt/terms#black_uc4",
        "https://w3id.org/caapt/terms#black_uc5",
        "https://w3id.org/caapt/terms#black_uc6",
        "https://w3id.org/caapt/terms#black_uc7",
        "https://w3id.org/caapt/terms#black_uc8",
        "https://w3id.org/caapt/terms#black_uc9",
        "https://w3id.org/caapt/terms#black_uc10",
        "https://w3id.org/caapt/terms#black_uc11",
        "https://w3id.org/caapt/terms#def_part",
        "https://w3id.org/caapt/terms#def_appellation",
        "https://w3id.org/caapt/terms#def_unknown",
        "https://w3id.org/caapt/terms#def_other",
        "https://w3id.org/caapt/terms#indian_uc2"
]
# ---------------------------------------------------------------------------

def read_entity_file(entity_path: str) -> Dict[str, int]:
    """
    Read a plain entities.txt (one URI per line) and return {uri: idx}.
    If file format is different, replace this loader accordingly.
    """
    d = {}
    with open(entity_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            uri = line.strip()
            if not uri:
                continue
            d[uri] = i
    return d

def build_allowed_tail_ids(entity_path_or_dict, allowed_uris=None) -> List[int]:
    """
    Map the hardcoded ALLOWED_TAIL_URIS (or provided allowed_uris) to integer ids.

    Accepts either:
      - entity_path_or_dict as a path string to entities.txt
      - OR a pre-loaded dict mapping entity_uri -> id
    """
    if allowed_uris is None:
        allowed_uris = ALLOWED_TAIL_URIS

    # Accept either dict or path
    if isinstance(entity_path_or_dict, dict):
        entity2id = entity_path_or_dict
    else:
        entity2id = read_entity_file(entity_path_or_dict)

    missing = []
    ids = []

    for uri in allowed_uris:
        if uri in entity2id:
            ids.append(entity2id[uri])
        else:
            # attempt simple variants (local name fallback)
            v1 = uri.strip("<>")
            v2 = uri.rstrip("/")
            v3 = uri.split("#")[-1]

            found = False
            for cand in (v1, v2, v3):
                if cand in entity2id:
                    ids.append(entity2id[cand])
                    found = True
                    break

            if not found:
                missing.append(uri)

    if missing:
        print(
            "Allowed tail URI(s) not found in entities mapping and will be ignored: %s",
            missing
        )

    if not ids:
        print(
            "build_allowed_tail_ids: no allowed tail ids were found. "
            "Downstream evaluation (shortlist) may produce no results."
        )

    return ids


def filter_triples_for_relation(triples: List[Tuple[str,str,str]], target_relation_uri: str) -> List[Tuple[str,str,str]]:
    """
    Keep only triples with relation exactly equal to target_relation_uri.
    (This implements Option 1: original-direction only.)
    """
    return [t for t in triples if t[1] == target_relation_uri]

def logits_to_shortlist_rank_and_topk(logits: torch.Tensor, allowed_tail_ids: List[int], true_label_global: int, topk: int = 10):
    """
    logits: 1-D tensor of length label_num (global entity logits)
    allowed_tail_ids: list[int] mapping shortlist positions -> global entity ids
    true_label_global: int, global entity id of the ground truth
    Returns: rank (int, 1-based in shortlist), topk_list [(global_id, score), ...]
    """
    if logits.dim() != 1:
        logits = logits.squeeze(0)
    device = logits.device
    allowed_tensor = torch.tensor(allowed_tail_ids, dtype=torch.long, device=device)
    # gather shortlist logits
    shortlist_logits = logits[allowed_tensor]  # shape [m]
    # compute sorted indices descending inside shortlist space
    vals, idxs = torch.sort(shortlist_logits, descending=True)
    # map shortlist index to global id
    shortlist_global_ids = [allowed_tail_ids[i] for i in idxs.cpu().tolist()]
    # find position of true_label_global inside allowed_tail_ids
    try:
        true_pos = allowed_tail_ids.index(true_label_global)
    except ValueError:
        # Shouldn't happen per your guarantee; return worst rank
        return len(allowed_tail_ids) + 1, [(gid, float(v)) for gid, v in zip(shortlist_global_ids, vals.cpu().tolist())]
    # find rank within shortlist (1-based)
    # compute rank by comparing shortlist logits
    true_score = shortlist_logits[true_pos]
    rank = int((shortlist_logits > true_score).sum().item()) + 1
    # prepare topk list
    tk = min(topk, len(allowed_tail_ids))
    topk_list = [(shortlist_global_ids[i], float(vals[i].cpu().item())) for i in range(tk)]
    return rank, topk_list

def compute_metrics_over_dataset(model, bert, tokenizer, device, input_ids_list, labels_list, entity_path, topk_report=10):
    """
    High-level helper: runs model over provided pre-built input_ids list (same format as get_data_from_rawdata)
    and computes MRR/hits@k restricted to the hardcoded shortlist. Returns aggregated metrics and prints sample topk.
    """
    allowed_ids = build_allowed_tail_ids(entity_path)
    allowed_set = set(allowed_ids)
    # mapping from global -> shortlist idx (optional)
    # allowed_index_map = {g: i for i, g in enumerate(allowed_ids)}

    device = device if isinstance(device, torch.device) else torch.device(device)
    model.to(device)
    model.eval()

    total = 0
    mrr_sum = 0.0
    hit_counts = {1:0, 3:0, 10:0}
    # For diagnostics, capture a small sample of top-k for first N
    sample_topk = []

    with torch.no_grad():
        for i, raw in enumerate(input_ids_list):
            logits = None
            inp = torch.tensor(raw).unsqueeze(0).to(device)
            # Build inputs_embeds in same way your notebook does
            word_embeds = bert.embeddings.word_embeddings(inp)
            pos_embeds = bert.embeddings.position_embeddings(torch.arange(0, inp.shape[1], dtype=torch.long).to(device))
            inputs_embeds = word_embeds + pos_embeds
            logits = model(inputs_embeds=inputs_embeds, input_ids=inp)  # [1, label_num]
            logits = logits.squeeze(0)  # [label_num]
            true_label = labels_list[i]
            if true_label not in allowed_set:
                # Per your guarantee this shouldn't happen; skip if it does
                # (Alternatively, you could treat as failure; here we skip)
                continue
            rank, topk_list = logits_to_shortlist_rank_and_topk(logits, allowed_ids, true_label, topk=topk_report)
            total += 1
            mrr_sum += 1.0 / rank
            if rank <= 1: hit_counts[1] += 1
            if rank <= 3: hit_counts[3] += 1
            if rank <= 10: hit_counts[10] += 1
            if len(sample_topk) < 5:
                sample_topk.append((i, rank, true_label, topk_list))

    if total == 0:
        raise RuntimeError("No evaluation examples after filtering or none had true labels in shortlist.")
    metrics = {
        "total": total,
        "MRR": mrr_sum / total,
        "Hit@1": hit_counts[1] / total,
        "Hit@3": hit_counts[3] / total,
        "Hit@10": hit_counts[10] / total
    }
    return metrics, sample_topk
