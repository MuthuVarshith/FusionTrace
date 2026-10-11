---
title: FusionTrace
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
startup_duration_timeout: 1h
pinned: false
license: mit
short_description: Privacy-first multimodal synthetic-media risk assessment
---

# FusionTrace

FusionTrace estimates how likely an image, audio clip or video is to be synthetic
or manipulated. Each detector contributes a risk signal; the report shows every
signal, its weight and how they were combined.

**Results are risk indicators, not proof of authenticity or manipulation.**

## Before you upload

- This is a demo running on shared Hugging Face hardware. Uploaded files are
  processed on this Space's server, kept for a limited retention period, and lost
  whenever the Space restarts.
- Only upload media you are allowed to share. Do not upload anything private or
  sensitive.
- The optional AI summary (when enabled by the Space owner) sends detector scores,
  not the media itself, to Google's Gemini API.

Source code, documentation and self-hosting instructions:
<https://github.com/MuthuVarshith/FusionTrace>
