# Interview notes: Qwen2.5-3B summarization with SFT + DPO (Gen Z news summaries)

Figures from training_summary.txt (sections 1-7). The generation results are from 8 test articles:
call them a qualitative check, not a benchmark.

---

## 1. 30-second pitch

> "I fine-tuned Qwen2.5-3B to summarize news, in two stages: supervised fine-tuning with LoRA,
> then preference tuning with DPO. I ran both on Vertex AI GPUs. Then I tried to make it
> summarize in Gen Z spoken style. No suitable preference dataset existed, so I built a 1,000-pair
> one with the Claude API, using automatic quality filters. The interesting result was that DPO
> scored 97% on held-out preference pairs but barely changed what the model actually wrote.
> I found the reason and fixed it with an SFT warm-up stage."

## 2. Two-minute version

1. **Goal.** Fine-tune a small open model for summarization and learn the whole post-training
   pipeline on a tight budget: SFT, then DPO, then evaluation.
2. **Pipeline.**
   - SFT: LoRA (rank 64) on CNN/DailyMail.
   - DPO: TRL's DPOTrainer with a second LoRA adapter on top of the merged SFT model. The
     reference model is the same model with the adapter switched off, so only one copy is
     needed on a 24 GB L4.
   - Infrastructure: a launcher that submits Vertex AI jobs in train / warm-up / inference
     modes, keeps results in GCS and sends Telegram progress updates. About $2 per DPO run.
3. **Debugging the first SFT run.** I audited it and found three problems:
   - 78% of examples had their summary cut off by the 512-token limit;
   - the loss covered the article and the padding, not just the summary;
   - early stopping was configured backwards.

   Fix: mask the prompt so the loss only counts the summary + end-of-sequence token, and
   truncate the article instead of the summary.
4. **The Gen Z dataset** (details in section 4 below). 1,000 pairs for ~$10:
   - Claude wrote the good summary plus two kinds of bad ones;
   - the CNN highlights served as a third, formal kind of bad summary;
   - automatic checks dropped about 20% of generations.
5. **The key finding.** DPO preference accuracy went from 88% to 97%, but the generated
   summaries had almost no slang (0.5 terms per summary). The log-probabilities showed why: DPO
   mainly pushed the bad summaries down; the good ones got slightly *less* likely
   (-110 -> -113). DPO picks between styles the model can already produce; it doesn't teach a
   new one.
6. **The fix.**
   - An SFT warm-up on the good summaries before DPO. With 3 epochs, slang went from 0 to
     1.75 terms per summary, with no invented numbers.
   - It also copied the data's habits: "lowkey" in 5 of 8 summaries, and a flippant tone on
     a death story.
   - Next: cap repeated slang in the data, and build bad summaries from the warm-up model's
     real mistakes before rerunning DPO.

## 3. Numbers to remember

| | |
|---|---|
| Model | Qwen2.5-3B base, LoRA (DPO adapter ~30 MB) |
| Dataset | 1,000 pairs (900 train / 100 val), ~$10 to generate |
| DPO eval | preference accuracy 88% -> 97%, reward margin 0.14 -> 1.37 |
| Slang per summary | base 0 -> DPO 0.5 -> 3-epoch warm-up 1.75 |
| Invented numbers | 0 in every model |
| Cost per run | ~$2 on one L4 GPU |

---

## 4. How the DPO dataset was built (build_genz_dpo.py)

Two steps: **generate** (call Claude) and **build** (filter, pair, split).

### Step 1: Generate

1. **Pick the articles.**
   - 1,260 CNN/DailyMail training articles from row 1,000 onward, so none overlap the SFT
     data (rows 0-999).
   - Each is cut to 640 Qwen tokens; articles under 150 words are skipped.
2. **Write the style guide** (the system prompt) from the two Hugging Face datasets:
   - genz-slang-pairs-1k: 60 slang examples showing the voice;
   - Human-Like-DPO-Dataset: 4 casual replies (good tone) and 4 stiff "As an AI..." replies
     (robotic tone). Replies with markdown were filtered out.

   Neither dataset is used as training pairs: the slang data has no news, and the Human-Like
   pairs are chit-chat.
3. **One Claude call per article.**
   - Settings: Claude Opus 5.5, low effort, structured JSON output; the Message Batches API
     halves the price, and the system prompt is cached.
   - Input: the article, its CNN highlights, and the highlights' word count as a length target.
     This fixed an early version that wrote 70+ word summaries.
   - Output:
     - `chosen`: a faithful Gen Z summary, 2-3 sentences;
     - `robotic`: the same facts in a stiff assistant voice;
     - `wrong_genz`: the same Gen Z voice with one fact changed or invented;
     - `sensitive`: a flag for deaths, crime and disasters, which get a toned-down voice.
   - Results are appended to `raw.jsonl`, and the batch ID is saved, so an interrupted run
     resumes without paying twice.

### Step 2: Build

4. **Check the good summary.** The article is dropped if `chosen` fails any of these:
   - **No invented numbers:** every number must be in the article, including spelled-out
     ones ("forty-eight" = 48).
   - **Enough slang:** at least 2 terms (1 for sensitive stories), counted with a
     hand-picked word list. The list learned from the slang dataset flagged ordinary words
     like "the" and "city", so I replaced it.
   - **Length:** at most 25 words over the highlights, at least 15 words; no ending on a
     question.
5. **Pick one bad summary per article.**
   - Quotas: 40% formal (the CNN highlights joined into normal sentences, not written by
     Claude), 30% robotic, 30% wrong Gen Z.
   - A hash of the article ID decides the type, so rebuilding gives the same dataset; if that
     type fails its checks, the next type is tried.
   - Checks:
     - the bad summary must be 0.7-1.3x the length of the good one, so DPO can't learn
       "shorter wins";
     - formal and robotic must contain no slang;
     - wrong Gen Z must contain at least one slang term, so it differs only in the fact.
6. **Split and save.**
   - 1,000 pairs, shuffled with a fixed seed, split 900 train / 100 val.
   - Each row has `prompt`, `chosen`, `rejected`, `rejected_kind`, `sensitive`, `id`.
   - Every dropped article and the reason go to `rejected_log.jsonl`.

### Result

- **Articles:** 1,260 generated -> 1,030 passed -> 1,000 kept.
- **Bad summary types:** 369 formal, 373 robotic, 258 wrong Gen Z.
- **Lengths:** good summaries average 51 words, bad ones 48 (no length shortcut).
- **Sensitive stories:** 433.
- **Dropped:** 230 too little slang, 8 invented numbers, 2 too long.
- **Cost:** ~$9.90 in Claude API usage.
- **Known weakness:** the good summaries repeat themselves ("lowkey" in ~48%, "ngl" in
  ~32%), and the warm-up model copied that habit.

### Design choices worth explaining

- **Why three kinds of bad summary?** Each one teaches one thing to avoid: too formal (the
  original news voice), too robotic (assistant voice), and wrong facts while sounding right.
  The wrong-fact type stops the model from learning "more slang always wins".
- **Why length matching?** In my first DPO run, the preferred and rejected summaries differed
  mostly in length and formatting. The model hit 100% accuracy within a third of an epoch by
  learning that shortcut. Matching lengths makes it learn style and facts.
- **Why automatic filters?** I can't read 1,260 generations. Cheap rule-based checks catch the
  errors that matter most (invented numbers, missing style), and the log of dropped articles
  shows what the generator gets wrong.

Spoken version:
> "No preference dataset existed for Gen Z news summaries, so I generated one. For each of
> about 1,260 news articles, Claude wrote a faithful Gen Z summary plus two bad versions: a
> robotic one and one with a single fact changed. The original CNN highlights served as a
> third, formal bad version. Rule-based filters dropped anything with invented numbers, too
> little slang or the wrong length, and I matched the lengths of good and bad summaries so DPO
> couldn't take a length shortcut. That gave 1,000 pairs for about $10 using the Batch API."

---

## 5. The SFT warm-up

### What it is and why

> "Before DPO, I fine-tuned the model with ordinary supervised training on the good Gen Z
> summaries. DPO can only choose between styles the model can already produce. My model had
> never written slang, so there was nothing for DPO to choose. The warm-up teaches the style
> first; DPO then refines it."

- DPO doesn't show the model a target to copy. It shifts probability, up for the good answer
  and down for the bad one, compared with a frozen copy of the starting model.
- If the good answer is very unlikely to begin with, DPO can win just by pushing the bad
  answers down. That's what happened: 97% accuracy, but the good summaries got slightly less
  likely and the outputs barely changed.

Pipeline:
```
Qwen2.5-3B -> SFT on CNN/DM -> SFT warm-up on the good Gen Z summaries -> DPO (good vs. bad)
```

The warm-up model is also DPO's **reference model**, so DPO measures improvement from a model
that already writes slang.

### How I implemented it

1. **Data** (dataset.py `encode_prompt_completion`, `load_jsonl_sft`):
   - Take `prompt` + `chosen` from the DPO file and append an end-of-sequence token.
   - Labels are `-100` on the prompt, so the loss only counts the summary.
   - Examples that are too long are skipped, not cut, so the "### Summary:" cue and the
     summary are never lost.
2. **Model** (model.py `get_lora_model(init_adapter)`):
   - `PeftModel.from_pretrained(base, adapter, is_trainable=True)` continues training the
     existing news-summary LoRA adapter instead of starting a new one.
   - The model keeps its summarizing skill and adds the style; the base weights stay frozen.
3. **Training** (train.py): new options `--jsonl`, `--val_jsonl`, `--completion_col`,
   `--init_adapter`, `--output`, `--epochs` and `--lr`. It's the same Hugging Face Trainer
   with early stopping on validation loss. With no options, the old CNN/DM training is
   unchanged.
4. **Cloud job** (vertex/submit_vertex.sh `MODE=warmup`):
   - Uploads the dataset and runs train.py from a chosen starting adapter.
   - Saves the adapter, the loss curves and a record of which adapter it started from.
   - A DPO job then uses it via `SFT_RUN=warmup-...`. Inference can also run on the warm-up
     alone, to check the style before DPO.

Spoken version:
> "I reused the DPO data: for each pair I took the prompt and the good summary as a
> supervised example, masked the prompt out of the loss and appended an end-of-sequence token
> so the model learns to stop. I loaded the existing LoRA adapter in trainable mode and kept
> training it, so the model kept its summarizing skill and picked up the style. It ran as a
> separate Vertex AI job, and its adapter became both the starting point and the reference
> model for DPO."

### Data and size

- **Not generated separately:** it's the DPO dataset (`dpo_data/genz/train.jsonl` /
  `val.jsonl`) using only `prompt` + `chosen`; `rejected` is ignored (`--completion_col chosen`).
- **Built in memory at training time:** prompt + summary + end-of-sequence token, with the
  prompt masked out of the loss. Nothing new is saved to disk.
- **Size:** 900 train / 100 val. The 3-epoch run sees each example up to 3 times; best
  checkpoint at step 450, early stopping at step 600.
- **Why reuse it:** no extra cost, and it teaches exactly the summaries DPO will prefer.
- **Trade-off:** warm-up and DPO share the same 900 prompts, and the warm-up copies the data's
  habits ("lowkey" overuse).

> "The warm-up data is the good side of my DPO pairs: 900 article prompts with their Gen Z
> summary, prompt masked out of the loss. No extra generation, and it teaches exactly the
> outputs DPO will prefer."

---

## 6. Results: warm-up vs. DPO

### During training

| Run | Metric | Result |
|---|---|---|
| Warm-up, 1 epoch, lr 5e-5 | validation loss | 1.73 -> 1.55 |
| Warm-up, 3 epochs, lr 1e-4 | best validation loss | 1.38 (step 450) |
| DPO on the 1-epoch warm-up | preference accuracy | 88% -> 97% |
| | reward margin | 0.14 -> 1.37 |
| | log-prob, good / bad summary | -110 -> -113 / -128 -> -144 |

The warm-up and DPO numbers measure different things and can't be compared directly.

### Generated summaries (8 test articles, greedy decoding)

| Model | Slang per summary | Avg words | Numbers kept | Invented numbers |
|---|---|---|---|---|
| Base Qwen | 0 | 74 | 21% | 0 |
| Warm-up, 1 epoch | 0 | 54 | 30% | 0 |
| DPO (on the 1-epoch warm-up) | 0.5 | 59 | 16% | 0 |
| **Warm-up, 3 epochs** | **1.75** | 57 | 28% | 0 |

The reference highlights average 50 words.

### What DPO did and didn't fix

- **In this run, DPO fixed nothing the 3-epoch warm-up can't do.** The warm-up is better on
  every measure.
- **Compared with its own starting point, DPO:**
  - made the voice more casual ("it's a bummer", "you gotta pre-order", "the real MVP");
  - but misused slang ("a wild ride, lowkey", "keep it lowkey" about a car crash);
  - added opinions ("a true legend", "a stark lesson");
  - kept fewer numbers (30% -> 16%).
- **Why:**
  - The starting model didn't write slang, so DPO lowered the bad summaries without raising
    the good ones; the freed probability went to filler and misused slang.
  - The bad summaries (formal, robotic, one fact changed) didn't cover the warm-up model's
    actual failures: overused slang, a flippant tone on tragedies, added opinions. The 258
    changed-fact pairs did not stop DPO dropping facts, possibly because one changed detail
    is a weak signal next to the style gap in the other 742 pairs.
- **What DPO should fix next time** (built on the 3-epoch warm-up): bad summaries that show
  overused slang, a flippant tone on sensitive stories, and changed or added facts. A warm-up
  can't teach these, because it only imitates; DPO can say "this specific thing is worse".

Spoken version:
> "In my first run, DPO didn't fix anything the warm-up couldn't. It made the voice more
> casual, but it also misused slang and added opinions, because the starting model didn't
> write slang yet and my bad examples didn't target the failures it actually made. The lesson:
> warm-up SFT gets the style in, and DPO removes specific failure modes, so the bad examples
> have to be built around the errors the warm-up model actually makes."

---

## 7. "Why SFT and DPO rather than prompting alone?"

What they're checking: that fine-tuning was a deliberate choice, not the default. Answer in
four parts: reasons, evidence, the honest gap, when prompting is better.

### Reasons (pick two or three)

- **It's a base model, not a chat model.** Qwen2.5-3B base isn't trained to follow
  instructions or to stop after answering. Prompting it gave 74-word summaries with **0 slang**,
  and its output had to be cut off with stop strings.
- **Small models follow style prompts badly.** A 3B model doesn't reliably hold a voice from
  instructions; training puts the style into the weights.
- **Cost and speed at inference.** The style guide given to Claude is ~60 slang examples plus
  tone examples: thousands of tokens on *every* call. The fine-tuned model needs only the short
  prompt, so each request is cheaper and faster, on one cheap GPU or on-device.
- **Consistency.** A trained model behaves the same way across thousands of articles; prompted
  output varies with the prompt's wording.
- **DPO teaches what to avoid.** A prompt can say "don't be robotic"; DPO trains directly
  against concrete bad examples (too formal, robotic, wrong facts).

### The point that makes the answer strong

**I did use prompting: to build the dataset.** A large model (Claude) with a detailed prompt
could write good Gen Z summaries. I used that to generate the training data, then fine-tuned a
much smaller model on it. That's **distillation**: pay the expensive model once (~$10), then
get similar behaviour from a model that's cheap to run.

### The honest gap (say it before they find it)

I haven't tested the strongest prompting baseline: **Qwen2.5-3B-Instruct with the full style
guide and a few examples in the prompt**. My "base" result is the base model with a one-line
instruction, which is a weak baseline. The fair comparison:

| | Prompting baseline | Fine-tuned |
|---|---|---|
| Model | Qwen2.5-3B-Instruct + style guide + few examples | base + SFT + warm-up + DPO |
| Compare | slang, numbers kept, invented facts, tone, prompt tokens per request | same |

### When prompting is the better choice

- You need it working today, or the requirements keep changing.
- Volume is low, so per-call cost doesn't matter.
- You can use a large API model.
- There's no budget for building training data.

Fine-tuning pays off when volume is high, the model must be small or private, or the behaviour
is hard to get from instructions.

### Model answer (~45 seconds)

> "Three reasons. First, I was working with a 3B base model. It isn't instruction-tuned, and
> prompting it gave zero slang and summaries that didn't stop, so prompting alone couldn't
> produce the behaviour. Second, cost: getting the style reliably needed a long style guide,
> thousands of tokens on every request, whereas a fine-tuned model needs only a one-line prompt
> and runs on one cheap GPU. Third, DPO lets me train directly against specific failures, like
> robotic tone or changed facts, which a prompt can only ask nicely to avoid.
>
> I didn't drop prompting, though. I used a large model with a detailed prompt to generate the
> training data, so it's really distillation into a small model. The baseline I haven't run yet
> is Qwen-3B-Instruct with the full style guide. That's the fair comparison, and it's my next
> experiment. If that baseline were close enough, prompting would be the simpler choice."

**Likely follow-up: "Why not just use Claude in production?"** Cost per request at volume,
speed, and control: a 3B model can run privately on your own hardware, and you own the
behaviour.

---

## 8. Likely follow-up questions

- **Why DPO and not RLHF/PPO?** It needs no separate reward model and no sampling during
  training, so it's cheap and stable; that suits a 3B model on a single GPU.
- **Isn't SFT then DPO the standard recipe?** Yes (InstructGPT, Zephyr). What I learned
  firsthand is *why*: DPO refines behaviour the model already has, and SFT is what puts it
  there.
- **Why not only SFT?** SFT only imitates the good answers and never shows what to avoid.
  DPO adds that signal.
- **How do you know the dataset is good?** Automatic filters, a log of every dropped example,
  balanced quotas, matched lengths, and reading samples by hand. Admit that it still needs a
  slang-variety cap.
- **How did you evaluate?** Training-time preference metrics, plus summaries generated side by
  side for base / SFT / DPO, scored for number recall, slang count and invented numbers.
  Weakness: only 8 articles and no human or LLM-judge scores. That's the next step.
- **What would you do differently?**
  - Run the SFT warm-up from the start.
  - Sample the bad summaries from the model itself ("on-policy"), so the pairs are harder.
  - Add an LLM judge for faithfulness and tone.
- **Biggest lesson?** A good training metric isn't the same as the behaviour you want. 97%
  preference accuracy looked like success until I read the outputs.

## 9. Avoid

- Saying "I built a Gen Z summarizer" as if it's finished. Say "I'm building one; here's what
  I learned".
- Hiding that I used Claude for the data and code. Describe it as a tool choice: synthetic
  data generation with automatic validation.
- Overclaiming from 8 test articles. Call them "qualitative checks".
