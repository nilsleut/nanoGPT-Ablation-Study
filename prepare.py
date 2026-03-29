"""
data/prepare.py -- Datenvorbereitung fuer nanoGPT.

Datenquellen in Reihenfolge (automatische Auswahl):
    1. FineWeb-Edu  (HuggingFace streaming, ~100M Tokens -- empfohlen)
    2. WikiText-103 (HTTP-Download, ~500MB entpackt)
    3. WikiText-2   (HTTP-Download, ~3MB -- nur fuer Tests)
    4. Synthetic    (lokal generiert, kein Download noetig -- Notfall-Fallback)

Aufruf:
    python data/prepare.py                     # auto
    python data/prepare.py --source=fineweb
    python data/prepare.py --source=wikitext103
    python data/prepare.py --source=wikitext2
    python data/prepare.py --source=synthetic  # offline, sofort

Output:
    data/train.bin    uint16 token array
    data/val.bin      uint16 token array
    data/meta.json    vocab_size, token counts
"""

import os, json, argparse, random
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent


def get_tokenizer():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


def encode_texts(texts, enc):
    ids = []
    for text in texts:
        chunk = enc.encode_ordinary(text)
        chunk.append(enc.eot_token)
        ids.extend(chunk)
    return np.array(ids, dtype=np.uint16)


def save_split(ids, fname):
    path = DATA_DIR / fname
    arr  = np.memmap(path, dtype=np.uint16, mode='w+', shape=(len(ids),))
    arr[:] = ids
    arr.flush()
    mb = path.stat().st_size / 1e6
    print(f"  {fname}: {len(ids):,} tokens  ({mb:.1f} MB)")


# ── 1. FineWeb-Edu ────────────────────────────────────────────

def prepare_fineweb(enc, n_docs=100_000):
    from datasets import load_dataset
    print(f"Downloading FineWeb-Edu ({n_docs:,} documents, streaming)...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu",
                      name="sample-10BT", split="train", streaming=True)
    texts = []
    for i, doc in enumerate(ds):
        if i >= n_docs: break
        texts.append(doc["text"])
        if (i + 1) % 10_000 == 0:
            print(f"  {i+1:,} / {n_docs:,}")
    print("Tokenising...")
    split = int(len(texts) * 0.995)
    return encode_texts(texts[:split], enc), encode_texts(texts[split:], enc)


# ── 2. WikiText-103 ───────────────────────────────────────────

def prepare_wikitext103(enc):
    from datasets import load_dataset
    print("Downloading WikiText-103 via HuggingFace (Salesforce/wikitext)...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    tr_texts = [d["text"] for d in ds["train"]      if d["text"].strip()]
    va_texts = [d["text"] for d in ds["validation"] if d["text"].strip()]
    print(f"  {len(tr_texts):,} train / {len(va_texts):,} val documents")
    print("Tokenising...")
    return encode_texts(tr_texts, enc), encode_texts(va_texts, enc)


# ── 3. WikiText-2 ─────────────────────────────────────────────

def prepare_wikitext2(enc):
    import urllib.request, zipfile
    url = ("https://s3.amazonaws.com/research.metamind.io/"
           "wikitext/wikitext-2-raw-v1.zip")
    zp  = DATA_DIR / "wikitext-2-raw-v1.zip"
    if not zp.exists():
        print("Downloading WikiText-2 (~3MB)...")
        urllib.request.urlretrieve(url, zp)
    with zipfile.ZipFile(zp) as z:
        tr = z.read("wikitext-2-raw/wiki.train.raw").decode()
        va = z.read("wikitext-2-raw/wiki.valid.raw").decode()
    tr_docs = [p for p in tr.split("\n\n") if p.strip()]
    va_docs = [p for p in va.split("\n\n") if p.strip()]
    tr_ids, va_ids = encode_texts(tr_docs, enc), encode_texts(va_docs, enc)
    print(f"  WARNING: WikiText-2 sehr klein ({len(tr_ids):,} tokens).")
    print("  Fuer echte Ablation wikitext103 oder fineweb verwenden.")
    return tr_ids, va_ids


# ── 4. Synthetic (offline, immer verfuegbar) ──────────────────

def prepare_synthetic(enc, n_tokens=5_000_000):
    """
    Generiert synthetische Trainingsdaten aus Wikipedia-Satzstrukturen.
    Kein Download noetig -- laeuft immer.
    Reicht fuer Ablation-Vergleiche (alle Runs haben dieselbe Datenbasis).
    NICHT fuer qualitative Sprachgenerierung geeignet.
    """
    import random
    print(f"Generating synthetic data ({n_tokens:,} tokens, no download needed)...")

    # Einfache englische Saetze als Vorlage
    templates = [
        "The {noun} {verb} the {noun2} in the {place}.",
        "Scientists have discovered that {noun} can {verb} under certain conditions.",
        "In {year}, researchers found a new method to {verb} {noun}.",
        "The study of {field} has revealed important insights about {topic}.",
        "{Name} argued that the relationship between {A} and {B} is {adj}.",
        "Recent advances in {field} suggest that {noun} plays a key role.",
        "The experiment showed that {adj} {noun} tends to {verb} more rapidly.",
        "According to the theory, {A} and {B} are fundamentally {adj}.",
        "This process involves the {noun} of several {noun2} components.",
        "Data from {year} indicates a significant change in {topic}.",
    ]
    nouns   = ["system","model","network","brain","cell","signal","layer","process",
               "function","structure","pattern","sequence","token","vector","matrix"]
    verbs   = ["represents","encodes","transforms","generates","predicts","processes",
               "activates","inhibits","modulates","integrates","computes","maps"]
    places  = ["cortex","layer","region","network","system","space","domain"]
    fields  = ["neuroscience","machine learning","linguistics","physics","biology"]
    topics  = ["attention","memory","language","vision","learning","computation"]
    adjs    = ["complex","hierarchical","distributed","sparse","dense","recurrent"]
    names   = ["The model","The network","This approach","The system","Our method"]
    years   = ["2018","2019","2020","2021","2022","2023"]

    random.seed(42)
    texts = []
    while True:
        t = random.choice(templates)
        t = t.replace("{noun}",  random.choice(nouns))
        t = t.replace("{noun2}", random.choice(nouns))
        t = t.replace("{verb}",  random.choice(verbs))
        t = t.replace("{place}", random.choice(places))
        t = t.replace("{field}", random.choice(fields))
        t = t.replace("{topic}", random.choice(topics))
        t = t.replace("{adj}",   random.choice(adjs))
        t = t.replace("{Name}",  random.choice(names))
        t = t.replace("{A}",     random.choice(nouns))
        t = t.replace("{B}",     random.choice(nouns))
        t = t.replace("{year}",  random.choice(years))
        texts.append(t)
        if len(texts) % 500 == 0:
            ids_so_far = sum(len(enc.encode_ordinary(tx)) for tx in texts[-500:])
            if len(texts) * (ids_so_far / 500) >= n_tokens:
                break

    print(f"  Generated {len(texts):,} sentences")
    split    = int(len(texts) * 0.99)
    tr_ids   = encode_texts(texts[:split], enc)
    va_ids   = encode_texts(texts[split:], enc)
    print(f"  WARNING: Synthetic data -- kein echtes Englisch.")
    print("  Loss wird niedrig sein, aber Sprachgenerierung ist sinnlos.")
    return tr_ids, va_ids


# ── Main ──────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source",
                   choices=["auto","fineweb","wikitext103","wikitext2","synthetic"],
                   default="auto")
    p.add_argument("--fineweb_docs", type=int, default=100_000)
    p.add_argument("--synthetic_tokens", type=int, default=5_000_000)
    args = p.parse_args()

    print("Loading GPT-2 tokenizer (gpt2)...")
    enc = get_tokenizer()
    print(f"  vocab_size={enc.n_vocab}, eot_token={enc.eot_token}")

    source = args.source
    error_log = []

    if source in ("auto", "fineweb"):
        try:
            train_ids, val_ids = prepare_fineweb(enc, args.fineweb_docs)
            source = "fineweb"
        except Exception as e:
            error_log.append(f"FineWeb: {e}")
            if source == "fineweb":
                raise
            print(f"FineWeb failed: {e}")
            source = "auto_wikitext103"

    if source == "auto_wikitext103":
        try:
            train_ids, val_ids = prepare_wikitext103(enc)
            source = "wikitext103"
        except Exception as e:
            error_log.append(f"WikiText-103: {e}")
            print(f"WikiText-103 failed: {e}")
            source = "auto_wikitext2"

    if source == "auto_wikitext2":
        try:
            train_ids, val_ids = prepare_wikitext2(enc)
            source = "wikitext2"
        except Exception as e:
            error_log.append(f"WikiText-2: {e}")
            print(f"WikiText-2 failed: {e}")
            print("All downloads failed. Using synthetic fallback.")
            train_ids, val_ids = prepare_synthetic(enc, args.synthetic_tokens)
            source = "synthetic"

    if source == "wikitext103":
        train_ids, val_ids = prepare_wikitext103(enc)
    elif source == "wikitext2":
        train_ids, val_ids = prepare_wikitext2(enc)
    elif source == "synthetic":
        train_ids, val_ids = prepare_synthetic(enc, args.synthetic_tokens)

    if len(val_ids) < 1000:
        print("WARNING: Val set sehr klein. Val-Loss wird unzuverlaessig sein.")

    print(f"\nSaving binary files (source={source})...")
    save_split(train_ids, "train.bin")
    save_split(val_ids,   "val.bin")

    meta = {
        "source":            source,
        "vocab_size":        enc.n_vocab,
        "vocab_size_padded": 50304,
        "train_tokens":      int(len(train_ids)),
        "val_tokens":        int(len(val_ids)),
        "encoding":          "gpt2",
    }
    with open(DATA_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nFertig.")
    print(f"  Train: {len(train_ids):,} tokens")
    print(f"  Val:   {len(val_ids):,} tokens")
    if error_log:
        print(f"  Errors encountered: {error_log}")
    print(f"\nNaechster Schritt: python train.py")

if __name__ == "__main__":
    main()