# Scientific Figure Captioning with PaliGemma

A multimodal captioning component developed for **PolySumm**, a [graduation-project prototype for scientific paper understanding](https://github.com/yassinelsharkawy/Graduation_Project). Given a scientific figure, the service generates a natural-language caption that can be used alongside extracted paper text and tables.

This repository contains a local PEFT adapter, the matching processor files, an inference API, and the notebook used to explore fine-tuning on the SciCap dataset.

## Why This Model?

Scientific papers communicate evidence through more than prose. Figures often contain the experimental results, trends, comparisons, or system structure that a text-only summarizer cannot fully preserve. PolySumm therefore treats figure understanding as a separate modality rather than dropping figures during PDF processing.

The model choice is grounded in **SCICAP: Generating Captions for Scientific Figures** by Ting-Yao Hsu, C. Lee Giles, and Ting-Hao Huang. The paper is relevant to this project for three reasons:

1. **The task matches the project.** SCICAP focuses on captions for real scientific figures rather than ordinary photographs. This makes it a closer fit for PolySumm's input than a general image-captioning dataset.
2. **The data reflects scientific writing.** SCICAP was constructed from arXiv computer-science and machine-learning papers. The paper describes figure extraction, figure-type classification, subfigure filtering, text normalization, and several caption-selection strategies, including a First Sentence collection.
3. **The paper identifies a real research challenge.** The authors report that generating reliable captions for scientific figures remains difficult, even with specialized baselines. We chose a SciCap-initialized PaliGemma checkpoint so that the project starts from a model exposed to this domain, while treating the generated captions as assistive summaries that still require evaluation and review.

The saved adapter in this directory is configured over:

```text
google/paligemma-3b-ft-scicap-224
```

The accompanying notebook documents a SciCap First-Sentence fine-tuning workflow and experiments with LoRA and memory-conscious training. The exact notebook contains multiple experimental configurations; the adapter metadata is the authoritative source for the saved inference artifact.

## What Is Included

| Path | Purpose |
| --- | --- |
| `app.py` | FastAPI inference service for one or more uploaded images |
| `paligemma-finetuned-caption/` | Saved LoRA adapter weights and configuration |
| `paligemma/processor/` | Local PaliGemma processor and tokenizer files |
| `cnn+lstm[final_final_working_paliGemma]-2.ipynb` | Fine-tuning and inference experiments |
| `2110.11624v2.pdf` | Local copy of the referenced SCICAP paper |

The adapter configuration records a LoRA adapter with rank `8`, alpha `8`, no bias, and target modules covering the attention projections and feed-forward projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.

## How It Fits Into PolySumm

```text
Scientific Paper PDF
    |
    +--> Figure extraction
              |
              +--> PaliGemma figure captioning
                            |
                            +--> Structured multimodal representation
                                          |
                                          +--> Retrieval and paper summarization
```

The captioning service is one component of PolySumm. It does not replace the text summarization, table summarization, or retrieval components.

## Setup

This service requires Python, PyTorch, and access to the Hugging Face model repository for the base checkpoint. A CUDA-capable GPU is recommended for practical inference; the code can fall back to CPU using `float32`, but the 3B-parameter base model may require substantial memory.

Install the packages used by the service:

```bash
pip install fastapi uvicorn python-multipart pillow torch transformers peft huggingface_hub
```

Set a Hugging Face token in the environment. Do not place tokens in source files, notebooks, or commits.

PowerShell:

```powershell
$env:HF_TOKEN = "hf_your_token_here"
```

From this directory, start the API:

```bash
uvicorn app:app --host localhost --port 8010
```

The interactive API documentation will be available at `http://localhost:8010/docs`.

## API Usage

Send one or more image files to `/captions`:

```bash
curl -X POST "http://localhost:8010/captions" \
  -F "files=@path/to/figure.png"
```

The response has this shape:

```json
{
  "captions": {
    "figure.png": "generated caption"
  }
}
```

The service converts images to RGB, uses the prompt `caption`, and generates text with the local adapter. The current inference settings use sampling and allow up to 512 new tokens. Generated captions can vary between runs and should be checked for factual accuracy, especially for axis labels, numerical values, equations, and dense multi-panel figures.

## Reproducibility Notes

- The local directory contains the adapter, not a complete standalone base model. The base PaliGemma checkpoint is downloaded by Transformers when it is not already cached.
- The processor files are stored locally under `paligemma/processor`.
- The notebook uses SciCap files and paths from its original Google Colab environment. Those paths must be adapted before reproducing training locally.
- The notebook records several experimental settings, including different data fractions and training configurations. This repository does not claim a single benchmark result from those experiments.
- No independently reproduced BLEU, accuracy, latency, or hardware comparison is asserted here. Such claims should be made only after fixing a dataset split, decoding configuration, evaluation script, and hardware setup.

## Limitations

This is a research prototype, not a certified scientific reporting system. The model may omit important visual details, misread text, produce plausible but unsupported statements, or struggle with equations, tables, small labels, and compound figures. Captions should therefore be treated as drafts or retrieval context, not as a replacement for the original figure and paper.

The model was developed around scientific figures represented in the SciCap setting. Performance on figures from other disciplines, unusual layouts, or low-quality scans may differ and should be evaluated separately.

## Reference

Hsu, T.-Y., Giles, C. L., & Huang, T.-H. K. (2021). **SCICAP: Generating Captions for Scientific Figures.** arXiv:2110.11624. [Paper](https://arxiv.org/abs/2110.11624)

## Relationship to the Main Project

This model is part of [PolySumm](../README.md), a multimodal scientific-paper summarization system combining document extraction, specialized text/figure/table processing, retrieval, and a user-facing application.
