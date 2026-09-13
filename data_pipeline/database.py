import sqlite3
import pandas as pd
from pathlib import Path


# Paths
BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "cleaned_books.csv"
DB_FILE = BASE_DIR / "zepto_books.db"


def create_database():
    # Read cleaned dataset
    df = pd.read_csv(CSV_FILE)

    # Connect to SQLite
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Drop tables if they already exist
    cursor.execute("DROP TABLE IF EXISTS books")
    cursor.execute("DROP TABLE IF EXISTS categories")

    # Create categories table
    cursor.execute("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        )
    """)

    # Create books table
    cursor.execute("""
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock BOOLEAN,
            category_id INTEGER,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
    """)

    # Insert unique categories
    categories = df["category"].dropna().unique()

    for category in categories:
        cursor.execute(
            "INSERT INTO categories (category_name) VALUES (?)",
            (category,)
        )

    # Create category lookup
    category_map = {}

    cursor.execute("SELECT category_id, category_name FROM categories")

    for category_id, category_name in cursor.fetchall():
        category_map[category_name] = category_id

    # Insert books
    for _, row in df.iterrows():

        category_id = category_map.get(row["category"])

        cursor.execute("""
            INSERT INTO books (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            row["title"],
            float(row["price_gbp"]),
            float(row["price_inr"]),
            int(row["rating"]),
            bool(row["in_stock"]),
            category_id
        ))

    conn.commit()

    # Verification
    cursor.execute("SELECT COUNT(*) FROM books")
    book_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM categories")
    category_count = cursor.fetchone()[0]

    print("Database created successfully!")
    print(f"Books inserted: {book_count}")
    print(f"Categories inserted: {category_count}")
    print(f"Database: {DB_FILE}")

    conn.close()


if __name__ == "__main__":
    create_database()