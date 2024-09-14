---
title: Case-Study-1 - Image-To-Music
emoji: 🎼
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

## Case-Study-1: Image-To-Music 🎼

An image to music converter, built with the following models:
- https://huggingface.co/Salesforce/blip-image-captioning-large for Image Captioning
- https://huggingface.co/microsoft/Phi-3-mini-4k-instruct       for Audio Prompt generation with Caption
- https://huggingface.co/facebook/musicgen-small                for Music Generation

Currently supports .jpg, .jpeg, and .png!