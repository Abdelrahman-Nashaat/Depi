# Clinical Note Intelligence

DEPI team project for classifying clinical notes and extracting structured information.

- `clinical.ipynb`: complete workflow, evaluation and dashboard.
- `app.py`: small Streamlit demo using the same prompts and API settings.

Install Python 3.12+ and the dependencies:

```bash
pip install -r requirements.txt
```

Open `clinical.ipynb` in Jupyter or VS Code. Run All uses the saved Groq evaluation
and trains the local baseline. The dataset downloads on the first run if needed.
Set `RUN_LIVE = True` for a new evaluation.

For the demo, add `GROQ_API_KEY=your_key` to a local `.env` file, then run:

```bash
streamlit run app.py
```

On Streamlit Community Cloud, select `app.py` and add the key in Secrets.
The demo needs Groq; the local fallback is available in the notebook.

The saved pilot scored 19/29 (65.5% accuracy, 0.598 macro F1). This is a small
sample, and extraction accuracy was not measured. Confidence is not accuracy.
