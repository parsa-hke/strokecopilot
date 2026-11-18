"""LoRA fine-tuning of the small reasoner on gold answers (Item 4, DEC-30). Logic for notebook 54; runs on a Colab A100.

- Base: Gemma 3 4B instruction-tuned, in bf16 (no quantisation, so the model served afterwards is the one trained).
  LoRA on the language model's attention and MLP projections only, never the vision tower.
- Data: EXP-43's train.jsonl / val.jsonl (train_data.py): each example is the evaluation request (system prompt v3
  with the rulebook, the case's user message) and the gold reply.
- Loss on the answer only. The chat template renders system + user + answer; the prompt part is the same render
  without the answer plus the generation prompt, and must be its prefix. The model returns logits for the last
  `answer_len + 1` positions only (`logits_to_keep`): Gemma's vocabulary is 262k tokens, so logits for the whole
  ~7,500-token request would take several GB that the answer-only loss never uses.
- Batch of one example, gradient accumulation, AdamW, cosine schedule with warmup; validation loss every
  `eval_every` optimiser steps, early stopping after `patience` evaluations without improvement; the best adapter is
  kept. A checkpoint (adapter, optimiser, scheduler, step) after each evaluation lets a disconnected Colab resume.
- merge(): base + best adapter -> one full model directory, served by vLLM with the same settings as the base.
- dummy(): a short run (Item 4, step 5) that reports peak GPU memory and tokens per second and checks that one
  generated answer is valid JSON for the schema.
"""

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

BASE_MODEL = "google/gemma-3-4b-it"
# Projections of the language model; the negative look-ahead keeps LoRA off the vision tower and its projector.
TARGET_MODULES = r"^(?!.*(vision|multi_modal)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


@dataclass
class FinetuneConfig:
    model_id: str = BASE_MODEL
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 20
    epochs: float = 2.0
    grad_accum: int = 8
    eval_every: int = 25  # optimiser steps
    patience: int = 3
    max_len: int = 10_000  # tokens; an example longer than this stops the run rather than being cut
    max_steps: int | None = None  # optimiser steps; overrides epochs (the dummy run)
    seed: int = 20261117
    gradient_checkpointing: bool = True
    notes: dict = field(default_factory=dict)


def read_jsonl(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def messages(system, example, with_answer=True):
    out = [{"role": "system", "content": system}, {"role": "user", "content": example["user"]}]
    if with_answer:
        out.append({"role": "assistant", "content": example["answer"]})
    return out


def _ids(tokenizer, msgs, add_generation_prompt):
    ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=add_generation_prompt)
    if isinstance(ids, dict) or hasattr(ids, "keys"):  # some tokenizers return a BatchEncoding
        ids = ids["input_ids"]
    return list(ids)


def encode(tokenizer, system, example, max_len):
    """(input_ids, answer_len): the full render, and how many of its final tokens are the answer (with the end-of-turn
    tokens the template adds after it). Raises if the prompt render is not a prefix of the full one, or if the example
    is longer than max_len."""
    prompt = _ids(tokenizer, messages(system, example, False), add_generation_prompt=True)
    full = _ids(tokenizer, messages(system, example, True), add_generation_prompt=False)
    if full[:len(prompt)] != prompt:
        raise ValueError(f"{example['id']}: the chat template does not render the prompt as a prefix of the full example.")
    if len(full) > max_len:
        raise ValueError(f"{example['id']}: {len(full)} tokens, over max_len {max_len}.")
    answer_len = len(full) - len(prompt)
    if answer_len < 1:
        raise ValueError(f"{example['id']}: no answer tokens.")
    return full, answer_len


def answer_loss(model, input_ids, answer_len):
    """Mean cross-entropy over the answer tokens of one example. input_ids: 1 x L tensor. Only the last
    answer_len + 1 positions get logits: position i predicts token i + 1, so the logits at L - answer_len - 1 .. L - 2
    predict the answer tokens L - answer_len .. L - 1."""
    import torch.nn.functional as F

    out = model(input_ids=input_ids, logits_to_keep=answer_len + 1, use_cache=False)
    logits = out.logits[:, :-1, :].float()
    target = input_ids[:, -answer_len:]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))


def lora_model(model, config):
    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                      target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, lora)


def load_base(model_id, device="cuda", dtype="bfloat16", token=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype), token=token,
                                                 attn_implementation="sdpa").to(device)
    return model, tokenizer


def _schedule(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@dataclass
class _State:
    step: int = 0  # optimiser steps done
    seen: int = 0  # examples consumed
    best: float = float("inf")
    since_best: int = 0
    history: list = field(default_factory=list)


def evaluate(model, encoded, device):
    import torch

    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for ids, answer_len in encoded:
            loss = answer_loss(model, torch.tensor([ids], device=device), answer_len)
            total += float(loss) * answer_len
            n += answer_len
    model.train()
    return total / max(1, n)


def train(model, tokenizer, system, train_rows, val_rows, out_dir, config, device="cuda", log=print):
    """Fine-tune `model` (a peft model) and keep the best adapter in out_dir/best. Resumes from out_dir/last when it
    exists. Returns the run record (also written to out_dir/train_log.json)."""
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    train_enc = [encode(tokenizer, system, r, config.max_len) for r in train_rows]
    val_enc = [encode(tokenizer, system, r, config.max_len) for r in val_rows]
    per_epoch = math.ceil(len(train_enc) / config.grad_accum)
    total = config.max_steps or max(1, int(per_epoch * config.epochs))
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: _schedule(s, total, config.warmup_steps))
    state = _State()
    last = out / "last"
    if (last / "state.json").exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(model, load_file(str(last / "adapter_model.safetensors")))
        optimiser.load_state_dict(torch.load(last / "optimiser.pt", map_location=device))
        scheduler.load_state_dict(torch.load(last / "scheduler.pt"))
        state = _State(**json.loads((last / "state.json").read_text()))
        log(f"Resuming after optimiser step {state.step} (best validation loss {state.best:.4f}).")
    order = []
    for epoch in range(math.ceil(total * config.grad_accum / len(train_enc)) + 1):
        idx = list(range(len(train_enc)))
        random.Random(config.seed + epoch).shuffle(idx)
        order += idx
    model.train()
    started, tokens, micro = time.time(), 0, 0
    optimiser.zero_grad(set_to_none=True)
    while state.step < total:
        ids, answer_len = train_enc[order[state.seen]]
        state.seen += 1
        loss = answer_loss(model, torch.tensor([ids], device=device), answer_len) / config.grad_accum
        loss.backward()
        tokens += len(ids)
        micro += 1
        if micro % config.grad_accum:
            continue
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimiser.step()
        scheduler.step()
        optimiser.zero_grad(set_to_none=True)
        state.step += 1
        if state.step % config.eval_every == 0 or state.step == total:
            val = evaluate(model, val_enc, device)
            elapsed = time.time() - started
            row = {"step": state.step, "examples": state.seen, "train_loss_last": float(loss) * config.grad_accum,
                   "val_loss": val, "lr": scheduler.get_last_lr()[0], "tokens_per_s": tokens / max(elapsed, 1e-9)}
            state.history.append(row)
            log(json.dumps(row))
            if val < state.best:
                state.best, state.since_best = val, 0
                model.save_pretrained(out / "best")
            else:
                state.since_best += 1
            _checkpoint(model, optimiser, scheduler, state, last)
            if state.since_best >= config.patience:
                log(f"Early stop at step {state.step}: no improvement in {config.patience} evaluations.")
                break
    record = {"config": asdict(config), "steps": state.step, "examples_seen": state.seen, "best_val_loss": state.best,
              "history": state.history, "train_examples": len(train_enc), "val_examples": len(val_enc),
              "tokens_per_example": {"mean": sum(len(i) for i, _ in train_enc) / len(train_enc),
                                     "max": max(len(i) for i, _ in train_enc)},
              "answer_tokens": {"mean": sum(a for _, a in train_enc) / len(train_enc)}}
    (out / "train_log.json").write_text(json.dumps(record, indent=1))
    return record


def _checkpoint(model, optimiser, scheduler, state, last):
    import torch

    last.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(last)
    torch.save(optimiser.state_dict(), last / "optimiser.pt")
    torch.save(scheduler.state_dict(), last / "scheduler.pt")
    (last / "state.json").write_text(json.dumps(asdict(state)))


def merge(model_id, adapter_dir, out_dir, token=None):
    """Base + adapter -> one full bf16 model directory (weights, config, tokenizer) for vLLM."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, token=token)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(model_id, token=token).save_pretrained(out_dir)
    try:  # Gemma 3 4B is multimodal: vLLM reads its processor config too
        AutoProcessor.from_pretrained(model_id, token=token).save_pretrained(out_dir)
    except (OSError, ValueError):
        pass
    return Path(out_dir)


def stop_ids(tokenizer):
    """The token IDs that end an answer: the tokenizer's end-of-sequence and, for Gemma, <end_of_turn> (the token the
    chat template puts after every answer, so the one training teaches). Passed to generate() explicitly rather than
    trusting the model's generation config."""
    ids = [tokenizer.eos_token_id]
    vocab = tokenizer.get_vocab()
    if "<end_of_turn>" in vocab:
        ids.append(vocab["<end_of_turn>"])
    return [i for i in dict.fromkeys(ids) if i is not None]


def generate_answer(model, tokenizer, system, example, max_new_tokens=3000, device="cuda", return_info=False):
    """Greedy answer for one example, decoded: the dummy run's check that the tuned model writes schema JSON. With
    return_info, (text, {"new_tokens", "stopped"}): stopped is False when the answer ran into max_new_tokens."""
    import torch

    prompt = _ids(tokenizer, messages(system, example, False), add_generation_prompt=True)
    stops = stop_ids(tokenizer)
    model.eval()
    with torch.no_grad():
        out = model.generate(input_ids=torch.tensor([prompt], device=device), max_new_tokens=max_new_tokens,
                             do_sample=False, eos_token_id=stops, pad_token_id=tokenizer.pad_token_id or stops[0])
    new = out[0][len(prompt):].tolist()
    text = tokenizer.decode(new, skip_special_tokens=True)
    if not return_info:
        return text
    return text, {"new_tokens": len(new), "stopped": bool(new) and new[-1] in stops}


def check_generated(text, vision_case_id="dummy"):
    """(valid, problem): whether a generated answer parses and validates as a recommendation (the run fields added)."""
    from strokecopilot.contract.validate import validate_recommendation

    try:
        reply = json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError as e:
        return False, f"not JSON: {e}"
    try:
        validate_recommendation({"case_id": vision_case_id, "model": "tuned", "prompt_version": "reasoner_v3", **reply})
    except Exception as e:  # ContractError
        return False, f"schema: {str(e)[:200]}"
    return True, None


def generation_check(model, tokenizer, system, rows, out_dir, n=3, max_new_tokens=1500, device="cuda"):
    """After the full run: the tuned model answers the first n validation rows. Validation loss cannot see a model that
    loops (the dummy run's 30-step model repeated one claim until the token cap), so this checks each answer stops by
    itself, validates against the schema and gives the gold verdicts. The longest gold answer is about 900 tokens, so
    max_new_tokens 1500 cuts only a loop. Writes generation_check.json and each answer as generated_<id>.txt."""
    out = Path(out_dir)
    results = []
    for row in rows[:n]:
        text, info = generate_answer(model, tokenizer, system, row, max_new_tokens, device, return_info=True)
        (out / f"generated_{row['id']}.txt").write_text(text, encoding="utf-8")
        valid, problem = check_generated(text)
        verdicts = None
        if valid:
            reply, gold_reply = json.loads(text.strip().removeprefix("```json").removesuffix("```").strip()), json.loads(row["answer"])
            verdicts = all(reply[k]["recommend"] == gold_reply[k]["recommend"] for k in ("thrombolysis", "thrombectomy"))
        results.append({"id": row["id"], "tokens": info["new_tokens"], "stopped": info["stopped"], "valid": valid,
                        "problem": problem, "verdicts_match_gold": verdicts})
    summary = {"n": len(results), "stopped": sum(r["stopped"] for r in results), "valid": sum(r["valid"] for r in results),
               "verdicts_match_gold": sum(bool(r["verdicts_match_gold"]) for r in results), "answers": results}
    (out / "generation_check.json").write_text(json.dumps(summary, indent=1))
    return summary


def dummy(model, tokenizer, system, train_rows, val_rows, out_dir, config, device="cuda", log=print):
    """Item 4 step 5: a few optimiser steps on a few examples, then one generated answer. Returns a report with peak
    GPU memory, tokens per second and whether the answer validates. Nothing from it is used for training."""
    import torch

    cuda = torch.cuda.is_available() and str(device).startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t = time.time()
    record = train(model, tokenizer, system, train_rows, val_rows, out_dir, config, device, log)
    seconds = time.time() - t
    text, info = generate_answer(model, tokenizer, system, val_rows[0], device=device, return_info=True)
    (Path(out_dir) / "generated.txt").write_text(text, encoding="utf-8")
    valid, problem = check_generated(text)
    report = {"steps": record["steps"], "examples_seen": record["examples_seen"], "seconds": round(seconds),
              "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1) if cuda else None,
              "tokens_per_s": record["history"][-1]["tokens_per_s"] if record["history"] else None,
              "tokens_per_example": record["tokens_per_example"], "answer_tokens": record["answer_tokens"],
              "val_loss": [h["val_loss"] for h in record["history"]], "generated_valid": valid,
              "generated_problem": problem, "generated_tokens": info["new_tokens"], "generated_stopped": info["stopped"],
              "generated_start": text[:300], "generated_end": text[-300:]}
    (Path(out_dir) / "dummy_report.json").write_text(json.dumps(report, indent=1))
    return report
