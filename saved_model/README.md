# saved_model/

Trained checkpoints and configuration produced by
[`notebooks/04_gridsearch_final_pipeline.ipynb`](../notebooks/04_gridsearch_final_pipeline.ipynb)
(its final "Save for API" section):

```
saved_model/
├── config.json          # Thresholds, ensemble weights, eval summary
├── AraBERT_best.pt        # Fine-tuned aubmindlab/bert-base-arabertv2 (best_epoch=6, MAE=3.29)
└── CAMeL_BERT_best.pt      # Fine-tuned CAMeL-Lab/bert-base-arabic-camelbert-mix (best_epoch=4, MAE=4.04)
```

`config.json` is real and versioned directly in this repo. The `.pt`
checkpoint files (~400–450 MB each) are tracked via **Git LFS** — see the
`*.pt` rule in the root [`.gitattributes`](../.gitattributes). Run
`git lfs pull` after cloning to fetch their actual content.

## `config.json` fields

| Field | Description |
|-------|--------------|
| `thresholds` | Score cut-offs (`T1`/`T2`/`T3`) separating the 4 suitability classes |
| `score_range` | Min/max raw score used to rescale predictions internally |
| `class_names` / `class_labels` | The 4 classification labels, in order |
| `winner_model` | Best single model by validation MAE (`AraBERT`) |
| `models` | Maps each model name to its HuggingFace Hub identifier |
| `ensemble_weights` | Weight given to each model when combining predictions |
| `eval_summary` | Validation MAE and best epoch per model |
| `fixed_params` | Training hyperparameters held constant during the grid search (see `JOURNEY.md`, Phase 5) |

## Loading the checkpoints

Each `.pt` file is a dict with `model_state`, `model_path`, `best_params`,
`thresholds`, `score_range`, `best_epoch`, and `best_mae` — load it with
the `CVRegressor` architecture (same as defined inline in the training
notebook):

```python
import torch
from transformers import AutoModel

ckpt = torch.load("saved_model/AraBERT_best.pt", map_location="cpu", weights_only=False)
# ckpt["model_path"] == "aubmindlab/bert-base-arabertv2"
# ckpt["model_state"] -> load into a CVRegressor instance
```

The same checkpoints are already deployed and served live at:
https://huggingface.co/spaces/omaraboelmaaty/Arabic_CV_Analyzer_API
