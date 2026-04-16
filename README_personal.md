# Why I Built FreshCheck

This is my personal research-style project to practice computer vision with a real problem I care about: **food waste and sustainability**.

I wanted to build something practical, not only a classifier:
- detect produce,
- estimate freshness stage,
- and suggest what to do next (eat now, wait, cook, or discard safely).

## Motivation

A lot of food is wasted because people are unsure if produce is still usable.  
I wanted to explore whether a simple AI assistant could reduce that uncertainty in everyday decisions.

## Why Multimodal Matters

Freshness is not only visual. People also use touch and smell.  
That is why I added a texture question in the app: a first step toward a human-in-the-loop multimodal approach.

## How I Worked

- Built fast to establish a working baseline.
- Used transfer learning (MobileNetV2) to leverage pretrained visual knowledge.
- Collected and cleaned data from multiple sources (web, personal captures, people I know).
- Iterated on fine-tuning and regularization for stability.

## What This Project Represents

This repo is a base for continuous work:
- better data collection (forms + manual cleaning),
- stronger label consistency,
- improved model robustness,
- and expansion to new produce classes (next: apples).

## Long-Term Vision

Keep evolving FreshCheck into an accessible, sustainability-focused AI tool while deepening my work in:
- computer vision,
- transfer learning,
- multimodal reasoning,
- and real-world ML for environmental impact.
