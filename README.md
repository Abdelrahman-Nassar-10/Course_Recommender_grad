# 🎓 Course Recommendation System

An interactive **course recommender web app** built with **Streamlit**.
It suggests similar courses based on a selected course title or user query, using **TF‑IDF embeddings + cosine similarity**.

## ✨ Features

* 🔎 **Smart Search with Autocomplete**
* 📚 **Content-based Recommendations** (cosine similarity over TF‑IDF vectors)
* 🎛️ **Filtering** by Subject, Difficulty, Rating, Price, Reviews, Year
* 🎚️ Optional **Quality‑aware ranking** (ratings + reviews)
* 🌀 **Diversification (MMR)** to avoid duplicates
* 📊 **Clean Streamlit UI** with course details and links

---

## 📂 Project Structure

```
course-recommender/
├── app.py                      # Streamlit app (main)
├── combined_courses.csv        # Dataset with courses
├── requirements.txt
├── README.md
└── data/                       # (optional) raw CSVs or notebooks
```

---

## 📦 Data

The app uses a CSV file (`combined_courses.csv`) with the following schema:

* `course_id`, `course_title`, `url`, `price`, `num_subscribers`, `num_reviews`,
  `num_lectures`, `level`, `Rating`, `content_duration`, `published_timestamp`, `subject`

⚠️ No external embedding files are needed — embeddings are built dynamically from the text fields using TF‑IDF.

---

## 🚀 Getting Started

### 1) Create & activate a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually [http://localhost:8501](http://localhost:8501)).

---

## 🧠 How it works (high level)

1. Loads `combined_courses.csv` into a pandas DataFrame.
2. Builds **TF‑IDF vectors** from `course_title + subject + level`.
3. Computes **cosine similarity** between the query course and candidate courses.
4. Applies optional **filters** (difficulty, rating, price, etc.).
5. Returns **Top‑N** recommendations with details & links.
6. Optional **MMR diversification** ensures the list is not full of duplicates.

---

## 🛠 Tech Stack

* Python 3.9+
* Streamlit
* pandas, numpy
* scikit-learn (TF‑IDF + cosine similarity)

---

## 👤 Author

**Abdelrahman Nassar** — [GitHub](https://github.com/Abdelrahman-Nassar-10)
