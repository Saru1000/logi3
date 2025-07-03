
# FastTrack Logistics – Streamlit Dashboard

AI‑powered logistics analytics demo built with **Streamlit** and **Plotly**.  
The app visualises a **synthetic survey dataset** (50 000 responses) that mimics
real‑world delivery‑company behaviour so you can explore:

* **Market viability** – demand heat‑maps & urgent‑delivery clusters  
* **Operational feasibility** – distance vs time regressions & AI‑route A/B sims  
* **Financial viability** – cost‑per‑km modelling & sensitivity analysis  
* **Competitive advantage** – churn/retention patterns & ROI calculators  

---

## 📁 Project structure

```
.
├── app.py                # Streamlit dashboard (entry‑point)
├── requirements.txt      # Python dependencies (Streamlit, Plotly, Pandas…)
├── generate_fasttrack_data.py  # Synthetic‑data generator (optional)
├── fasttrack_survey_synthetic.csv  # 50k‑row sample dataset  (~15 MB)
└── README.md             # You’re here
```

> **Hint:** If you only need to *run* the dashboard, keep `app.py`,  
> `requirements.txt` and the CSV. The generator script is optional.

---

## 🚀 Quick start (local)

```bash
# 1 Clone the repo
git clone https://github.com/<you>/fasttrack-logistics.git
cd fasttrack-logistics

# 2 Create & activate a virtual env  (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3 Install dependencies
pip install -r requirements.txt

# 4 Launch the dashboard
streamlit run app.py
```

Open the URL printed in your terminal (usually <http://localhost:8501>)  
and start exploring 📈.

---

## 🌐 Deploy on Streamlit Cloud

1. Push the project to a public/private GitHub repo.  
2. Go to **streamlit.io → New app** and connect your repo.  
3. Set **main file** to `app.py` & **branch** to *main* (or your branch).  
4. Click **Deploy** – Streamlit Cloud installs `requirements.txt` automatically.

> **Tweak:** In *Advanced settings → Secrets* you can store API keys or set
> `GENERATE_DATA=0` if you don’t want to re‑generate the CSV on every rebuild.

---

## 🔄 Regenerating the synthetic dataset (optional)

If you want a **fresh dataset** with a different random seed or row‑count:

```bash
python generate_fasttrack_data.py  # default 50 000 rows
#  or
python generate_fasttrack_data.py --rows 100000 --seed 123
```

The script outputs `fasttrack_survey_synthetic.csv` in the project root.

---

## 🛠️ Customisation tips

* **Modify visuals** – open `app.py`; each tab corresponds to one feasibility
  dimension. Plotly charts are built in small helper functions.
* **Add models** – drop new ML code in `/models` and import it inside
  `app.py` (remember to pin new libs in `requirements.txt`).
* **Data volume** – for quick demos, sample 10 k rows; for stress‑tests,
  bump to 1 M rows (mind memory!).

---

## 📚 Tech stack

| Layer       | Package |
|-------------|---------|
| App runner  | Streamlit |
| Viz         | Plotly Express, Altair |
| Data        | Pandas, NumPy |
| ML (optional)| Scikit‑learn, mlxtend |

---

## 📖 License

MIT – do whatever you want, just give attribution.  
© 2025 FastTrack Logistics demo by **YOUR NAME**.

---

*Happy streaming & speedy deliveries! 🚚*
