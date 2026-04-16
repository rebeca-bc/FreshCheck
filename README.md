# FreshCheck AI

FreshCheck is a two-stage computer vision project for produce freshness:
1. **Stage 1** identifies the produce type.
2. **Stage 2** predicts freshness stage with a produce-specific model.

The app then translates predictions into practical guidance (days remaining, action, recipes, storage tips) plus an environmental impact converter.

## Live Demo

**Deploy link:** `[https://your-deploy-link-here](https://freshcheck-ripeness.streamlit.app/)`  

<img width="1440" height="809" alt="Screen Shot 2026-04-15 at 23 31 22" src="https://github.com/user-attachments/assets/b56eb68c-edaf-42cd-9cf7-0a43a194eada" />


## Highlights

- Two-stage hierarchical classification (transfer learning, MobileNetV2).
- 5 produce categories (avocado, banana, spinach, strawberry, tomato).
- Interactive texture question (human-in-the-loop signal).
- Days-remaining estimate + recommendations.
- Recipe suggestions + storage guidance.
- Banana-bread callout for overripe bananas.
- Waste-impact converter (quantity -> estimated CO2e and water footprint).

## Repository Structure

```text
projectClassifier/
├── app.py                     # root launcher (streamlit run app.py)
├── app/
│   └── app.py                 # main Streamlit app
├── train_model_stage1.py
├── train_model_stage2.py
├── predict_stage1.py
├── augment_data.py
├── ripeness_database.py
├── requirements.txt
├── README.md
└── README_personal.md
```

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes for GitHub Posting

- Training graph images are ignored via `.gitignore` (as requested).
- Local env/cache files are ignored.
- If you do not want to publish raw data later, add `data/` and `data_stages/` to `.gitignore`.

## Technical Direction

- Transfer learning baseline inspired by recent maturity-detection research workflows.
- Ongoing work: label normalization, class balance improvements, more data collection, and expansion to additional produce (e.g., apples).

---
Created by **Rebeca Borrego Cavazos**.
