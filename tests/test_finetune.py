"""LoRA fine-tuning (finetune.py) on CPU with a tiny random Gemma 3 text model and a character tokenizer whose chat
template renders like Gemma's. No network, no GPU."""

import json
import re

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from strokecopilot.nlp.reasoner import finetune as ft  # noqa: E402

TEMPLATE = ("{{ bos_token }}{% for m in messages %}{% if m['role'] == 'system' %}{% set sys = m['content'] %}"
            "{% elif m['role'] == 'user' %}<u>{{ sys }}|{{ m['content'] }}</u>"
            "{% else %}<m>{{ m['content'] }}</m>{% endif %}{% endfor %}{% if add_generation_prompt %}<m>{% endif %}")


@pytest.fixture(scope="module")
def tok():
    from tokenizers import Tokenizer, models, pre_tokenizers

    chars = sorted(set('abcdefghijklmnopqrstuvwxyz0123456789 {}[]":,.|<>/_-ABCDEFGHIJKLMNOPQRSTUVWXYZ\n'))
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, **{c: i + 4 for i, c in enumerate(chars)}}
    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    t = transformers.PreTrainedTokenizerFast(tokenizer_object=core, bos_token="<bos>", eos_token="<eos>",
                                             pad_token="<pad>", unk_token="<unk>")
    t.chat_template = TEMPLATE
    return t


def _model(tok, seed=0):
    torch.manual_seed(seed)
    cfg = transformers.Gemma3TextConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                        num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=512,
                                        sliding_window=64, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    return transformers.Gemma3ForCausalLM(cfg)


SYSTEM = "rules"
ROWS = [{"id": f"t{i}", "user": f"case {i}", "answer": '{"a": 1}'} for i in range(6)]


def test_encode_marks_the_answer_and_refuses_long_examples(tok):
    ids, n = ft.encode(tok, SYSTEM, ROWS[0], max_len=200)
    assert tok.decode(ids[-n:]) .replace(" ", "") == '{"a":1}</m>'.replace(" ", "")
    with pytest.raises(ValueError, match="over max_len"):
        ft.encode(tok, SYSTEM, ROWS[0], max_len=10)


def test_answer_loss_equals_cross_entropy_on_full_logits(tok):
    model = _model(tok)
    ids, n = ft.encode(tok, SYSTEM, ROWS[0], max_len=200)
    x = torch.tensor([ids])
    full = model(input_ids=x).logits[0]
    expected = torch.nn.functional.cross_entropy(full[len(ids) - n - 1:len(ids) - 1].float(), x[0, -n:])
    assert float(ft.answer_loss(model, x, n)) == pytest.approx(float(expected), rel=1e-5)


def test_lora_targets_the_language_model_only():
    assert re.fullmatch(ft.TARGET_MODULES, "model.language_model.layers.3.self_attn.q_proj")
    assert re.fullmatch(ft.TARGET_MODULES, "model.layers.0.mlp.down_proj")
    assert not re.fullmatch(ft.TARGET_MODULES, "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj")
    assert not re.fullmatch(ft.TARGET_MODULES, "model.multi_modal_projector.mm_input_projection")


def test_train_lowers_the_loss_keeps_the_best_and_resumes(tok, tmp_path):
    model = ft.lora_model(_model(tok), ft.FinetuneConfig(lora_r=4, lora_alpha=8, lora_dropout=0.0))
    cfg = ft.FinetuneConfig(lr=5e-2, warmup_steps=1, grad_accum=2, eval_every=2, patience=10, max_steps=6,
                            max_len=200, gradient_checkpointing=False)
    rec = ft.train(model, tok, SYSTEM, ROWS, ROWS[:2], tmp_path, cfg, device="cpu", log=lambda *_: None)
    losses = [h["val_loss"] for h in rec["history"]]
    assert rec["steps"] == 6 and losses[-1] < losses[0]
    assert (tmp_path / "best" / "adapter_model.safetensors").exists() and (tmp_path / "last" / "state.json").exists()
    lines = []
    cfg.max_steps = 8
    rec2 = ft.train(model, tok, SYSTEM, ROWS, ROWS[:2], tmp_path, cfg, device="cpu", log=lines.append)
    assert lines[0].startswith("Resuming after optimiser step 6") and rec2["steps"] == 8


def test_merge_writes_a_model_that_loads(tok, tmp_path):
    base = _model(tok)
    base.save_pretrained(tmp_path / "base")
    tok.save_pretrained(tmp_path / "base")
    peft_model = ft.lora_model(_model(tok), ft.FinetuneConfig(lora_r=4, lora_alpha=8))
    peft_model.save_pretrained(tmp_path / "adapter")
    out = ft.merge(str(tmp_path / "base"), tmp_path / "adapter", tmp_path / "merged")
    merged = transformers.AutoModelForCausalLM.from_pretrained(out)
    assert sum(p.numel() for p in merged.parameters()) == sum(p.numel() for p in base.parameters())


def test_check_generated_accepts_schema_json_only():
    from strokecopilot.nlp.reasoner import gold
    from strokecopilot.nlp.verifier import scenarios as sc

    s = sc.generate()[0]
    good = json.dumps(gold.gold_answer(s.vision, s.facts))
    assert ft.check_generated(good) == (True, None) and ft.check_generated("```json\n" + good + "\n```")[0]
    assert not ft.check_generated("not json")[0] and "schema" in ft.check_generated('{"thrombolysis": 1}')[1]


def test_dummy_reports_speed_loss_and_whether_the_answer_validates(tok, tmp_path, monkeypatch):
    model = ft.lora_model(_model(tok), ft.FinetuneConfig(lora_r=4, lora_alpha=8, lora_dropout=0.0))
    cfg = ft.FinetuneConfig(max_steps=3, grad_accum=1, eval_every=1, warmup_steps=1, max_len=200, gradient_checkpointing=False)
    monkeypatch.setattr(ft, "generate_answer", lambda *a, **k: ("not json", {"new_tokens": 3000, "stopped": False}))
    report = ft.dummy(model, tok, SYSTEM, ROWS, ROWS[:2], tmp_path, cfg, device="cpu", log=lambda *_: None)
    assert report["steps"] == 3 and len(report["val_loss"]) == 3 and report["tokens_per_s"] > 0
    assert report["generated_valid"] is False and report["peak_gpu_gb"] is None
    assert report["generated_stopped"] is False and report["generated_tokens"] == 3000
    assert (tmp_path / "generated.txt").read_text() == "not json"
    assert json.loads((tmp_path / "dummy_report.json").read_text())["steps"] == 3


def test_generation_stops_on_end_of_turn_and_reports_whether_it_stopped(tok):
    assert ft.stop_ids(tok) == [tok.eos_token_id]  # no <end_of_turn> in this vocabulary
    gemma_like = transformers.PreTrainedTokenizerFast(tokenizer_object=tok.backend_tokenizer, eos_token="<eos>")
    gemma_like.add_special_tokens({"additional_special_tokens": ["<end_of_turn>"]})
    assert ft.stop_ids(gemma_like) == [gemma_like.eos_token_id, gemma_like.convert_tokens_to_ids("<end_of_turn>")]
    text, info = ft.generate_answer(_model(tok), tok, SYSTEM, ROWS[0], max_new_tokens=5, device="cpu", return_info=True)
    assert isinstance(text, str) and 1 <= info["new_tokens"] <= 5 and isinstance(info["stopped"], bool)
    assert info["stopped"] == (info["new_tokens"] < 5)


def test_generation_check_reports_stops_validity_and_verdicts(tok, tmp_path, monkeypatch):
    from strokecopilot.nlp.reasoner import gold
    from strokecopilot.nlp.verifier import scenarios as sc

    s = sc.generate()[0]
    good = json.dumps(gold.gold_answer(s.vision, s.facts))
    flipped = json.loads(good)
    flipped["thrombolysis"]["recommend"] = not flipped["thrombolysis"]["recommend"]
    rows = [{"id": "a", "answer": good}, {"id": "b", "answer": good}, {"id": "c", "answer": good}, {"id": "d", "answer": good}]
    replies = iter([(good, {"new_tokens": 600, "stopped": True}), (json.dumps(flipped), {"new_tokens": 600, "stopped": True}),
                    ('{"claims": [1, 1, 1', {"new_tokens": 1500, "stopped": False})])
    monkeypatch.setattr(ft, "generate_answer", lambda *a, **k: next(replies))
    summary = ft.generation_check(None, tok, SYSTEM, rows, tmp_path, n=3, device="cpu")
    assert (summary["n"], summary["stopped"], summary["valid"], summary["verdicts_match_gold"]) == (3, 2, 2, 1)
    assert [r["verdicts_match_gold"] for r in summary["answers"]] == [True, False, None]
    assert (tmp_path / "generated_c.txt").read_text().startswith('{"claims"')
    assert json.loads((tmp_path / "generation_check.json").read_text())["stopped"] == 2
