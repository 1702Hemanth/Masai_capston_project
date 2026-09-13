# Module 1 — Data Pipeline

## Objective

This module implements a complete data pipeline for scraping, cleaning,
enriching, storing, and querying book catalogue data from Books to Scrape.

Pipeline:

Scrape → Clean → Convert → Store → Query → Pandas Validation

---

## Data Source

Source website:

https://books.toscrape.com/

The project uses the first 5 paginated pages of the All Products catalogue.

The raw scrape produces 100 books.

After cleaning, 14 rows with unparseable/unknown category values were
dropped, leaving 86 valid book records.

The final dataset contains 27 different book categories.

---

## Technologies Used

- Python
- requests
- BeautifulSoup
- pandas
- SQLite
- sqlite3

---

## Installation

From the project root:

```bash
pip install -r requirements.txt

