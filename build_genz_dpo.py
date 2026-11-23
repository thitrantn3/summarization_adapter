"""
Build a ~1k-pair DPO dataset for Gen Z-style news summaries, using Claude to write them.

Sources
  - cnn_dailymail 3.0.0 (train rows after the ones SFT used): the news articles + highlights
  - Programmer-RD-AI/genz-slang-pairs-1k: slang examples for the style guide, and the word
    list used to check that a summary actually sounds Gen Z
  - HumanLLMs/Human-Like-DPO-Dataset: tone references only (its casual "chosen" replies set
    the voice; its stiff "As an AI..." "rejected" replies model the robotic rejected summary)

One Claude call per article returns four texts:
  chosen        faithful Gen Z summary
  formal        (not generated) the CNN/DM highlights themselves
  robotic       stiff, assistant-like summary of the same facts
  wrong_genz    Gen Z summary with one fact changed or invented

Each kept article becomes one pair. The rejected kind is assigned by quota
(40% formal / 30% robotic / 30% wrong_genz), falling back to another kind when the
assigned one fails a check.

Prompt = the SFT template plus a style instruction:
  "Summarize the following news article in Gen Z slang:\n\n{article}\n\n### Summary:\n"

Steps (both run by default; --step lets you run one):
  generate   call Claude, append raw results to <out_dir>/raw.jsonl (resumable: done ids are skipped)
  build      filter raw.jsonl and write <out_dir>/train.jsonl, val.jsonl, rejected_log.jsonl

Usage
  pip install anthropic datasets transformers
  export ANTHROPIC_API_KEY=...
  python build_genz_dpo.py --limit 5            # try 5 articles first and read the output
  python build_genz_dpo.py                      # full run (~1,150 articles -> ~1,000 pairs)
  python build_genz_dpo.py --batch              # same, via the Batch API: half price, results in < 24 h
                                                # (usually < 1 h); Ctrl-C is safe, rerun --batch to resume
  python build_genz_dpo.py --step build         # re-filter without calling the API
  python train_dpo.py --train_file dpo_data/genz/train.jsonl --val_file dpo_data/genz/val.jsonl
"""
import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from tqdm import tqdm

from config import APITOKEN, MODEL_NAME

INSTRUCTION = "Summarize the following news article in Gen Z slang:"
MAX_ARTICLE_TOKENS = 640   # Qwen tokens; leaves room for a ~120-token summary inside train_dpo's 1,024
MIN_ARTICLE_WORDS = 150
REJECTED_QUOTA = {"formal": 0.4, "robotic": 0.3, "wrong_genz": 0.3}

# Word list for the "does this sound Gen Z?" check. Hand-picked: only terms that are
# unambiguous slang. Words that are also normal news vocabulary ("fire", "dead", "sick",
# "ate", "wild", "lit", "tea", "cap") are left out, or only counted inside a phrase,
# otherwise a plain news summary would pass as Gen Z.
SLANG_WORDS = set("""
bruh bestie bussin chillin chillin' chillax crib deadass delulu drip fam fave finna fr
goated hella homie hyped iykyk lowkey highkey low-key high-key nah peep peeped peeps
periodt rizz rn sesh sheesh simp slay slayed stan sus tryna vibe vibes vibin vibin'
vibing y'all yeet yo lol ngl tbh smh istg glow-up rent-free deets tho straight-up
clapped
""".split())
SLANG_PHRASES = [
    "no cap", "it's giving", "main character", "understood the assignment", "touch grass",
    "big yikes", "spill the tea", "the tea is", "took an l", "took the l", "big w",
    "hits different", "say less", "on god", "caught in 4k", "ate and left no crumbs",
    "straight fire", "living rent free", "rent free", "for the plot", "spilled the tea",
    "dip out", "dipped out", "pulled up", "pull up",
]
# "took the W", "a massive L", "another W": capital W/L after a cue word. Bare W/L is not
# counted, since news has "George W. Bush" and "L.A.".
WL_RE = re.compile(r"\b(?i:the|a|an|big|massive|major|huge|easy|another|took|taking|take|takes)\s+[WL]\b"
                   r"|\b\d+-\d+\s+[WL]\b"  # "a 1-0 W"
                   r"|\b[WL]\s+(?i:energy)\b")

SYSTEM_PROMPT = """You write training data for a small model that summarizes news in Gen Z slang.

The user sends a news article and its official highlights, with a word count for the
highlights. Return JSON with these fields:

"chosen": the GOOD summary.
  - LENGTH: about as long as the highlights: aim for their word count, and never more than
    10 words over it. 2-3 short sentences. Cover the same main points the highlights cover;
    leave out the other details. Count your words before answering.
  - Plain text: no bullets, no hashtags, at most one emoji.
  - Every fact must come from the article (the highlights are a guide to what matters).
    Keep the key names, places and numbers exactly as written there; never add a number,
    name or detail that is not in the article.
  - Casual Gen Z voice: about 1-3 slang terms per sentence, mixed naturally, varied across
    articles. Sound like a real person telling a friend the news, not a parody.
  - The slang examples below lean hard on "Yo", "vibing" and "chillin'". Use them rarely;
    don't open with "Yo", and draw on the wider Gen Z vocabulary (lowkey, ngl, no cap,
    it's giving, the tea, big W / took an L, hits different, sus, rizz, bestie, fr, etc.).
    Vary your openers and fillers too; don't put "ngl", "lowkey", "fr" or "real talk" in every summary.
  - No questions to the reader, no "What about you?", no talking about yourself.
  - If the story involves death, serious injury, abuse, disasters or crime victims, set
    "sensitive" to true and keep the voice casual but respectful: no jokes, no "slay",
    "no cap", "it's giving" or emoji about the harm itself.

"robotic": a BAD summary of the same facts in a stiff, formal, assistant-like voice
  (like the robotic examples below): "This article reports that...", "It is important to
  note...". Same length as "chosen", no slang.

"wrong_genz": a BAD summary in the same Gen Z voice and length as "chosen", but with
  exactly one key fact wrong: change a number, swap a name or place, flip an outcome, or
  invent a detail. It must read as confident and plausible.

"changed_fact": one short sentence saying what you changed in "wrong_genz".

"sensitive": true or false, as described above.

## Gen Z slang examples (plain -> Gen Z)
{slang_examples}

## Casual human voice examples (tone to aim for in "chosen", but stay on the news)
{casual_examples}

## Robotic voice examples (tone for "robotic")
{robotic_examples}
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "chosen": {"type": "string"},
        "robotic": {"type": "string"},
        "wrong_genz": {"type": "string"},
        "changed_fact": {"type": "string"},
        "sensitive": {"type": "boolean"},
    },
    "required": ["chosen", "robotic", "wrong_genz", "changed_fact", "sensitive"],
    "additionalProperties": False,
}

# Gen Z summary may be at most this many words longer than the CNN highlights. Loose on
# purpose: the rejected side is length-matched separately (0.7-1.3x) in check_rejected.
MAX_WORDS_OVER_HIGHLIGHTS = 25
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
MARKDOWN_RE = re.compile(r"\*\*|^\s*(?:\d+\.|[-*]) ", re.M)
WORD_RE = re.compile(r"[a-z][a-z'\-]*")


# ---------------------------------------------------------------- style sources

def first_words(text, n):
    words = text.split()
    return " ".join(words[:n]) + (" ..." if len(words) > n else "")


def load_style_sources(seed, n_slang, n_tone):
    rng = random.Random(seed)
    slang = load_dataset("Programmer-RD-AI/genz-slang-pairs-1k", split="train")
    human = load_dataset("HumanLLMs/Human-Like-DPO-Dataset", split="train")

    # Spread the slang examples over the whole set (it repeats a few templates a lot)
    seen, examples = set(), []
    for i in rng.sample(range(len(slang)), len(slang)):
        key = " ".join(slang[i]["normal"].lower().split()[:3])
        if key in seen:
            continue
        seen.add(key)
        examples.append(f"- {slang[i]['normal']} -> {slang[i]['gen_z']}")
        if len(examples) == n_slang:
            break

    # Tone examples: skip replies with markdown (bold, lists) so Claude doesn't copy the formatting
    plain_idx = [i for i in rng.sample(range(len(human)), min(len(human), 500))
                 if not MARKDOWN_RE.search(human[i]["chosen"] + human[i]["rejected"])]
    idx = plain_idx[:n_tone]
    casual = [f"- {first_words(human[i]['chosen'], 45)}" for i in idx]
    robotic = [f"- {first_words(human[i]['rejected'], 45)}" for i in idx]

    system = SYSTEM_PROMPT.format(
        slang_examples="\n".join(examples),
        casual_examples="\n".join(casual),
        robotic_examples="\n".join(robotic),
    )
    return system


def slang_count(text):
    """Number of slang words + slang phrases in text (see SLANG_WORDS / SLANG_PHRASES)."""
    words = WORD_RE.findall(text.lower().replace("’", "'"))
    joined = " " + " ".join(words) + " "
    return (sum(w in SLANG_WORDS for w in words)
            + sum(joined.count(f" {p} ") for p in SLANG_PHRASES)
            + len(WL_RE.findall(text.replace("\u2019", "'"))))


# ---------------------------------------------------------------- articles

def load_articles(start, num):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=APITOKEN)
    data = load_dataset("cnn_dailymail", "3.0.0", split="train")
    rows = []
    i = start
    while len(rows) < num and i < len(data):
        ex = data[i]
        i += 1
        if len(ex["article"].split()) < MIN_ARTICLE_WORDS:
            continue
        ids = tok(ex["article"].strip(), add_special_tokens=False)["input_ids"][:MAX_ARTICLE_TOKENS]
        rows.append({
            "id": ex["id"],
            "article": tok.decode(ids, skip_special_tokens=True).strip(),
            "highlights": ex["highlights"].strip(),
        })
    return rows


def make_prompt(article):
    return f"{INSTRUCTION}\n\n{article}\n\n### Summary:\n"


# ---------------------------------------------------------------- generate

class AccountError(Exception):
    """Billing / auth problem: every call will fail the same way, so stop the run."""


def request_params(args, system, article, highlights):
    """Request body shared by the live and batch paths."""
    return dict(
        model=args.model,
        max_tokens=4000,
        output_config={"effort": args.effort,
                       "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": (
            f"News article:\n\n{article}\n\n"
            f"Highlights ({len(highlights.split())} words):\n{highlights}")}],
    )


def parse_response(response):
    """Turn a Message into the saved row fields (or an {"error": ...} dict)."""
    if response.stop_reason == "refusal":
        return {"error": "refusal"}
    if response.stop_reason == "max_tokens":
        return {"error": "max_tokens"}
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        return {"error": "no_text"}
    out = json.loads(text)
    out["usage"] = {"input": response.usage.input_tokens,
                    "output": response.usage.output_tokens,
                    "cache_read": response.usage.cache_read_input_tokens or 0,
                    "cache_write": response.usage.cache_creation_input_tokens or 0}
    out["model"] = response.model
    return out


def call_claude(client, args, system, article, highlights):
    import anthropic

    try:
        response = client.beta.messages.create(
            **request_params(args, system, article, highlights),
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # if the model declines (e.g. on a violent story), retry on a fallback model
        )
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
        raise AccountError(e.message) from e
    except anthropic.BadRequestError as e:
        if "credit balance" in str(e.message).lower():
            raise AccountError(e.message) from e
        return {"error": f"bad_request: {e.message}"}
    except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
        return {"error": f"retryable: {type(e).__name__}"}  # SDK already retried; rerun later to resume
    return parse_response(response)


def pending_articles(args, raw_path):
    done = set()
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            for line in f:
                r = json.loads(line)
                if "error" not in r or r["error"].startswith(("bad_request", "refusal")):
                    done.add(r["id"])  # don't retry permanent failures; retryable ones run again
    articles = [a for a in load_articles(args.start, args.num) if a["id"] not in done]
    if args.limit:
        articles = articles[: args.limit]
    print(f"{len(done)} articles already done, {len(articles)} to generate with {args.model}")
    return articles


def report(rows):
    errors, tokens = Counter(), Counter()
    for out in rows:
        if "error" in out:
            errors[out["error"].split(":")[0]] += 1
        else:
            tokens.update(out["usage"])
    print(f"tokens: {dict(tokens)}  errors: {dict(errors) or 'none'}")


def generate(args, system):
    import anthropic

    raw_path = os.path.join(args.out_dir, "raw.jsonl")
    articles = pending_articles(args, raw_path)
    if not articles:
        return

    client = anthropic.Anthropic(max_retries=5)
    lock = threading.Lock()
    outs = []
    with open(raw_path, "a") as f, ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(call_claude, client, args, system, a["article"], a["highlights"]): a for a in articles}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="claude"):
            a = futures[fut]
            try:
                out = fut.result()
            except AccountError as e:
                pool.shutdown(wait=True, cancel_futures=True)
                raise SystemExit(f"Stopped: {e}\nNothing was saved for this article; "
                                 "rerun after fixing it and the run resumes.")
            outs.append(out)
            with lock:
                f.write(json.dumps({**a, **out}) + "\n")
                f.flush()
    report(outs)


def generate_batch(args, system):
    """Same requests through the Message Batches API (50% cheaper, results within 24 h).

    The batch id is saved in <out_dir>/batch.json, so if you stop the script (Ctrl-C) or it
    crashes, rerunning with --batch picks up the same batch instead of paying for a new one.
    """
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    raw_path = os.path.join(args.out_dir, "raw.jsonl")
    state_path = os.path.join(args.out_dir, "batch.json")
    client = anthropic.Anthropic(max_retries=5)

    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        print(f"resuming batch {state['batch_id']} ({len(state['articles'])} articles)")
    else:
        articles = pending_articles(args, raw_path)
        if not articles:
            return
        # Batches don't accept `fallbacks`, so a refused article is just logged and skipped.
        requests = [Request(custom_id=a["id"],
                            params=MessageCreateParamsNonStreaming(**request_params(args, system, a["article"], a["highlights"])))
                    for a in articles]
        try:
            batch = client.messages.batches.create(requests=requests)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.BadRequestError) as e:
            raise SystemExit(f"Batch not created: {e.message}")
        state = {"batch_id": batch.id, "model": args.model,
                 "articles": {a["id"]: a for a in articles}}
        with open(state_path, "w") as f:
            json.dump(state, f)
        print(f"submitted batch {batch.id} with {len(requests)} requests "
              f"(id saved in {state_path}; Ctrl-C is safe, rerun --batch to resume)")

    batch_id = state["batch_id"]
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        print(f"[{time.strftime('%H:%M:%S')}] {batch.processing_status}: {c.succeeded} succeeded, "
              f"{c.errored} errored, {c.processing} processing", flush=True)
        if batch.processing_status == "ended":
            break
        time.sleep(args.poll_seconds)

    outs = []
    with open(raw_path, "a") as f:
        for result in client.messages.batches.results(batch_id):
            a = state["articles"].get(result.custom_id)
            if a is None:
                continue
            kind = result.result.type
            if kind == "succeeded":
                out = parse_response(result.result.message)
            elif kind == "errored" and result.result.error.error.type == "invalid_request_error":
                out = {"error": f"bad_request: {result.result.error.error.message}"}
            else:  # server error, expired or canceled: retried on the next run
                out = {"error": f"retryable: batch_{kind}"}
            outs.append(out)
            f.write(json.dumps({**a, **out}) + "\n")
    os.remove(state_path)
    report(outs)


# ---------------------------------------------------------------- build

_UNITS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
          "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
NUMBER_WORDS = {w: i for i, w in enumerate(_UNITS)}
for _t, _w in enumerate(_TENS, start=2):
    NUMBER_WORDS[_w] = _t * 10
    for _u in range(1, 10):
        NUMBER_WORDS[f"{_w}-{_UNITS[_u]}"] = _t * 10 + _u
NUMBER_WORDS.update({"a dozen": 12, "dozen": 12, "hundred": 100, "thousand": 1000})
NUMBER_WORD_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, NUMBER_WORDS), key=len, reverse=True)) + r")\b", re.I)


def numbers(text, spelled=False):
    """Digits in text; with spelled=True also numbers written as words ("Forty-eight" -> 48)."""
    found = {n.replace(",", "").rstrip(".") for n in NUM_RE.findall(text)}
    if spelled:
        found |= {str(NUMBER_WORDS[m.lower()]) for m in NUMBER_WORD_RE.findall(text)}
    return found


def words(text):
    return len(text.split())


def check_chosen(r):
    """Return None if the chosen summary is usable, else the reason it isn't."""
    c = r["chosen"].strip()
    limit = words(r["highlights"]) + MAX_WORDS_OVER_HIGHLIGHTS
    if not 15 <= words(c) <= limit:
        return f"chosen_length_{words(c)}_limit_{limit}"
    invented = numbers(c) - numbers(r["article"], spelled=True)
    if invented:
        return f"chosen_invented_numbers_{sorted(invented)}"
    need = 1 if r.get("sensitive") else 2
    if slang_count(c) < need:
        return "chosen_not_genz"
    if c.rstrip().endswith("?"):
        return "chosen_asks_question"
    return None


def clean_highlights(text):
    """CNN highlights are one fragment per line ending in " ."; make them normal sentences so
    the formal rejected side differs from chosen in style only, not in formatting."""
    lines = [re.sub(r"\s+\.$", "", ln.strip()).rstrip(".") for ln in text.splitlines() if ln.strip()]
    lines = [re.sub(r"^NEW:\s*", "", ln) for ln in lines]
    return " ".join(ln[0].upper() + ln[1:] + "." for ln in lines if ln)


def check_rejected(kind, r):
    chosen = r["chosen"].strip()
    text = clean_highlights(r["highlights"]) if kind == "formal" else r[kind]
    text = text.strip()
    if not text or text == chosen:
        return None, "empty_or_same"
    ratio = words(text) / max(1, words(chosen))
    if not 0.7 <= ratio <= 1.3:
        return None, f"length_ratio_{ratio:.2f}"
    if kind in ("formal", "robotic") and slang_count(text) > 0:
        return None, "has_slang"
    if kind == "wrong_genz" and slang_count(text) < 1:
        return None, "not_genz"
    return text, None


def assign_kind(article_id):
    """Deterministic quota assignment, so re-running build gives the same split."""
    x = int(hashlib.md5(article_id.encode()).hexdigest(), 16) % 1000 / 1000
    cum = 0.0
    for kind, share in REJECTED_QUOTA.items():
        cum += share
        if x < cum:
            return kind
    return kind


def build(args):
    raw_path = os.path.join(args.out_dir, "raw.jsonl")
    rows = {}
    with open(raw_path) as f:
        for line in f:
            r = json.loads(line)
            rows[r["id"]] = r  # later lines (retries) win

    pairs, log, reasons, kinds = [], [], Counter(), Counter()
    for r in rows.values():
        if "error" in r:
            reasons[f"api_{r['error'].split(':')[0]}"] += 1
            continue
        bad = check_chosen(r)
        if bad:
            reasons[re.sub(r"_[\d\[].*$", "", bad)] += 1  # group "chosen_length_95" under "chosen_length"
            log.append({"id": r["id"], "reason": bad, "chosen": r["chosen"]})
            continue
        first = assign_kind(r["id"])
        order = [first] + [k for k in REJECTED_QUOTA if k != first]
        rejected = None
        for kind in order:
            rejected, why = check_rejected(kind, r)
            if rejected:
                break
            log.append({"id": r["id"], "reason": f"{kind}_{why}"})
        if not rejected:
            reasons["no_usable_rejected"] += 1
            continue
        kinds[kind] += 1
        pairs.append({
            "prompt": make_prompt(r["article"]),
            "chosen": r["chosen"].strip(),
            "rejected": rejected,
            "rejected_kind": kind,
            "sensitive": r.get("sensitive", False),
            "id": r["id"],
        })

    pairs.sort(key=lambda p: p["id"])
    random.Random(args.seed).shuffle(pairs)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    n_val = max(1, int(len(pairs) * args.val_frac))
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}
    for name, split in splits.items():
        with open(os.path.join(args.out_dir, f"{name}.jsonl"), "w") as f:
            for p in split:
                f.write(json.dumps(p) + "\n")
    with open(os.path.join(args.out_dir, "rejected_log.jsonl"), "w") as f:
        for entry in log:
            f.write(json.dumps(entry) + "\n")

    print(f"{len(rows)} articles -> {len(pairs)} pairs "
          f"({len(splits['train'])} train / {len(splits['val'])} val)")
    print(f"rejected kinds: {dict(Counter(p['rejected_kind'] for p in pairs))}")
    print(f"sensitive stories: {sum(p['sensitive'] for p in pairs)}")
    print(f"dropped: {dict(reasons) or 'none'}  (details in rejected_log.jsonl)")
    if pairs:
        cw = sum(words(p["chosen"]) for p in pairs) / len(pairs)
        rw = sum(words(p["rejected"]) for p in pairs) / len(pairs)
        print(f"avg words: chosen {cw:.0f}, rejected {rw:.0f}")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["all", "generate", "build"], default="all")
    ap.add_argument("--start", type=int, default=1000, help="first CNN/DM train row (SFT used 0-999)")
    ap.add_argument("--num", type=int, default=1150, help="articles to generate for (extra covers filtering)")
    ap.add_argument("--limit", type=int, default=None, help="only generate this many new articles (for a test run)")
    ap.add_argument("--max_pairs", type=int, default=1000)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--out_dir", default="./dpo_data/genz")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", action="store_true",
                    help="use the Message Batches API: 50%% cheaper, results within 24 h (usually < 1 h)")
    ap.add_argument("--poll_seconds", type=int, default=60, help="how often --batch checks the batch status")
    ap.add_argument("--n_slang_examples", type=int, default=60)
    ap.add_argument("--n_tone_examples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show_system", action="store_true", help="print the system prompt and exit")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.show_system:
        print(load_style_sources(args.seed, args.n_slang_examples, args.n_tone_examples))
        return
    if args.step in ("all", "generate"):
        system = load_style_sources(args.seed, args.n_slang_examples, args.n_tone_examples)
        (generate_batch if args.batch else generate)(args, system)
    if args.step in ("all", "build"):
        build(args)


if __name__ == "__main__":
    main()
