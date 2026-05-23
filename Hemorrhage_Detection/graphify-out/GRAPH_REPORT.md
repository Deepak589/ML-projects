# Graph Report - Hemorrhage_Detection  (2026-05-23)

## Corpus Check
- 20 files · ~1,224,580 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 193 nodes · 235 edges · 18 communities (14 shown, 4 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `9d701b87`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]

## God Nodes (most connected - your core abstractions)
1. `get_dataloaders()` - 13 edges
2. `build_model()` - 13 edges
3. `Hemorrhage Detection — Modernization Design` - 13 edges
4. `HemorrhageDataset` - 10 edges
5. `predict_image()` - 10 edges
6. `File Structure` - 10 edges
7. `load_model()` - 9 edges
8. `Task 1: Archive legacy files and scaffold project` - 9 edges
9. `FocalLoss` - 8 edges
10. `Intracranial Hemorrhage Detection` - 8 edges

## Surprising Connections (you probably didn't know these)
- `test_get_dataloaders_returns_three_loaders()` --calls--> `get_dataloaders()`  [EXTRACTED]
  tests/test_data.py → src/data.py
- `test_train_loader_batch_shape()` --calls--> `get_dataloaders()`  [EXTRACTED]
  tests/test_data.py → src/data.py
- `test_build_model_returns_module()` --calls--> `build_model()`  [EXTRACTED]
  tests/test_model.py → src/model.py
- `test_model_forward_shape()` --calls--> `build_model()`  [EXTRACTED]
  tests/test_model.py → src/model.py
- `tmp_checkpoint()` --calls--> `build_model()`  [EXTRACTED]
  tests/test_predict.py → src/model.py

## Communities (18 total, 4 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (32): code:bash (git add src/config.py), code:python (import pytest), code:python ("""Dataset, transforms, splits, dataloaders."""), code:bash (git add src/data.py tests/test_data.py), code:python (import torch), code:python ("""Model factory."""), code:bash (git add src/model.py tests/test_model.py), code:python ("""Training entry point: AMP, early stopping, ROC-AUC checkp) (+24 more)

### Community 1 - "Community 1"
Cohesion: 0.13
Nodes (19): Dataset, get_dataloaders(), get_eval_transforms(), get_train_transforms(), HemorrhageDataset, _make_weighted_sampler(), Dataset, transforms, splits, dataloaders., Loads PNG CT scans from class-named subfolders. (+11 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (22): Architecture, code:block1 (HemorrhageDetection/), code:python (def load_model(checkpoint_path: str, device: str = "auto") -), code:block3 (torch>=2.1), Components, Data Flow, Dataset, Dependencies (`requirements.txt`) (+14 more)

### Community 3 - "Community 3"
Cohesion: 0.15
Nodes (15): build_model(), Create a timm model with a custom classification head., evaluate(), FocalLoss, main(), Training entry point: AMP, early stopping, ROC-AUC checkpoint., Multiclass focal loss operating on logits + integer labels., set_seed() (+7 more)

### Community 4 - "Community 4"
Cohesion: 0.13
Nodes (14): code:block1 (src/        # package: config, data, model, train, evaluate,), code:bash (python -m venv .venv), code:bash (python -m src.train --epochs 30 --batch-size 32), code:bash (python -m src.evaluate --checkpoint checkpoints/best_model.p), code:python (from src.predict import load_model, predict_image), code:bash (pytest -v), Configuration, Evaluate (+6 more)

### Community 5 - "Community 5"
Cohesion: 0.15
Nodes (12): accuracy, f1_binary, f1_macro, n_test, pr_auc, precision_binary, precision_macro, recall_binary (+4 more)

### Community 6 - "Community 6"
Cohesion: 0.29
Nodes (11): load_model(), predict_image(), Inference API for Streamlit (or any external caller)., Load a trained checkpoint and return (model, class_names)., Predict the class of a single image.      tta: average over flip/rotation views, _resolve_device(), _to_ndarray(), test_load_model_returns_module_and_classnames() (+3 more)

### Community 7 - "Community 7"
Cohesion: 0.27
Nodes (10): best_threshold(), collect_predictions(), main(), plot_confusion(), plot_roc(), Evaluation entry point: metrics, confusion matrix, ROC curve., Return (probs_of_hemorrhage, labels). Preds derived later via threshold., Pick threshold maximizing F-beta (beta>1 favors recall) on given split. (+2 more)

### Community 8 - "Community 8"
Cohesion: 0.22
Nodes (9): code:bash (mkdir -p legacy src tests checkpoints outputs), code:bash (git mv Main.py legacy/Main.py), code:markdown (# Legacy), code:block4 (torch>=2.1), code:python, code:python, code:bash (touch checkpoints/.gitkeep outputs/.gitkeep), code:bash (git add legacy/ src/ tests/ checkpoints/ outputs/ requiremen) (+1 more)

### Community 9 - "Community 9"
Cohesion: 0.43
Nodes (4): close(), graph(), predict(), trainYolo()

### Community 10 - "Community 10"
Cohesion: 0.29
Nodes (6): backend, class_name, config, layers, name, keras_version

### Community 11 - "Community 11"
Cohesion: 0.33
Nodes (5): code:block31, code:block32, code:bash (git add README.md), {"label": "hemorrhage", "confidence": 0.92, "probs": {"normal": 0.08, "hemorrhage": 0.92}}, Verification Checklist (end of plan)

## Knowledge Gaps
- **79 isolated node(s):** `allow`, `class_name`, `name`, `layers`, `keras_version` (+74 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `File Structure` connect `Community 0` to `Community 8`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Why does `get_dataloaders()` connect `Community 1` to `Community 3`, `Community 7`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **Why does `build_model()` connect `Community 3` to `Community 6`, `Community 7`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **What connects `allow`, `class_name`, `name` to the rest of the system?**
  _97 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.0625 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.13 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.08695652173913043 - nodes in this community are weakly interconnected._