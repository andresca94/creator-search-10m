from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass(frozen=True)
class CrossEncoderConfig:
    model_name: str
    batch_size: int = 16
    max_length: int = 256
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    num_threads: int = 2  # CPU only
    log_every_batches: int = 25


def _resolve_device(device: str) -> str:
    d = (device or "auto").lower()
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d in ("cuda", "cpu"):
        return d
    return "cpu"


# Cache: include batch_size too (cfg is frozen; key must reflect inference behavior)
# key = (model_name, resolved_device, max_length, batch_size)
_CE_CACHE: Dict[Tuple[str, str, int, int], "CrossEncoderReranker"] = {}


class CrossEncoderReranker:
    def __init__(self, cfg: CrossEncoderConfig):
        self.cfg = cfg
        dev = _resolve_device(cfg.device)
        self.device = torch.device(dev)

        if self.device.type == "cpu":
            try:
                torch.set_num_threads(max(int(cfg.num_threads), 1))
            except Exception:
                pass

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        if not pairs:
            return []

        out_scores: List[float] = []
        bs = max(int(self.cfg.batch_size), 1)

        total = len(pairs)
        n_batches = (total + bs - 1) // bs
        log_every = max(int(self.cfg.log_every_batches), 1)

        t0 = time.perf_counter()

        for b, i in enumerate(range(0, total, bs), start=1):
            batch = pairs[i : i + bs]
            queries = [q for (q, _d) in batch]
            docs = [d for (_q, d) in batch]

            enc = self.tokenizer(
                queries,
                docs,
                padding=True,
                truncation=True,
                max_length=int(self.cfg.max_length),
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            logits = self.model(**enc).logits  # [B, 1] or [B, num_labels]

            if logits.shape[-1] == 1:
                vals = logits.squeeze(-1)
                out_scores.extend([float(x) for x in vals.detach().cpu().tolist()])
            else:
                vals = logits.max(dim=-1).values
                out_scores.extend([float(x) for x in vals.detach().cpu().tolist()])

            if (b % log_every == 0) or (b == 1) or (b == n_batches):
                dt = time.perf_counter() - t0
                done = min(i + bs, total)
                print(
                    f"[CE] {done}/{total} pairs | batch {b}/{n_batches} | {dt:.1f}s | device={self.device.type}",
                    flush=True,
                )

        return out_scores


def get_cross_encoder(cfg: CrossEncoderConfig) -> CrossEncoderReranker:
    dev = _resolve_device(cfg.device)
    key = (cfg.model_name, dev, int(cfg.max_length), int(cfg.batch_size))
    ce = _CE_CACHE.get(key)
    if ce is None:
        ce = CrossEncoderReranker(
            CrossEncoderConfig(
                model_name=str(cfg.model_name),
                batch_size=int(cfg.batch_size),
                max_length=int(cfg.max_length),
                device=str(dev),
                num_threads=int(cfg.num_threads),
                log_every_batches=int(cfg.log_every_batches),
            )
        )
        _CE_CACHE[key] = ce
    return ce


def _extract_source(hit: Dict[str, Any]) -> Dict[str, Any]:
    # Most common shape in your pipeline: {"_source": {...}}
    src = hit.get("_source")
    if isinstance(src, dict) and src:
        return src

    # Fallbacks if some wrappers exist
    for k in ("_source_doc", "_raw_source", "_os_source", "source"):
        inner = hit.get(k)
        if isinstance(inner, dict) and inner:
            return inner
    return {}


def _hit_to_doc_text(hit: Dict[str, Any]) -> str:
    src = _extract_source(hit)

    bio = str(src.get("bio") or "")
    keywords = " ".join([str(x) for x in (src.get("keywords") or [])])
    verticals = " ".join([str(x) for x in (src.get("verticals") or [])])
    recent_text = str(src.get("recent_text") or "")

    text = " ".join([verticals, keywords, bio, recent_text]).strip()
    return text[:3000]


def rerank_with_cross_encoder(
    query: str,
    hits: List[Dict[str, Any]],
    top_m: int,
    ce: CrossEncoderReranker,
    alpha: float = 1.0,
) -> List[Dict[str, Any]]:
    if not hits:
        return hits

    m = min(max(int(top_m), 0), len(hits))
    head = hits[:m]
    tail = hits[m:]

    pairs = [(query, _hit_to_doc_text(h)) for h in head]
    ce_scores = ce.score_pairs(pairs)

    a = float(alpha)
    blended: List[Dict[str, Any]] = []

    for h, ce_s in zip(head, ce_scores):
        base = float(h.get("final_score") or 0.0)
        new_score = (1.0 - a) * base + a * float(ce_s)

        h2 = dict(h)
        h2["ce_score"] = float(ce_s)
        h2["final_score"] = float(new_score)

        exp = h2.get("explain")
        if isinstance(exp, dict):
            exp2 = dict(exp)
            exp2["ce_score"] = float(ce_s)
            h2["explain"] = exp2

        blended.append(h2)

    blended.sort(key=lambda x: float(x.get("final_score") or 0.0), reverse=True)
    return blended + tail
