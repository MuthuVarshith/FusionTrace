"""LoRA fine-tuning of the default image detector (Community Forensics ViT-S/16-384).

Modules:
    data.py   manifests (path, label, group, split), disjointness checks, datasets
    model.py  build the base ViT, apply LoRA to attention layers, predict scores

Entry points: ``backend/scripts/finetune_lora.py`` and ``backend/scripts/evaluate_finetune.py``.
See ``docs/how-to-finetune.md``.
"""
