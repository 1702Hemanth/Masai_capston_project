import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_FILE = BASE_DIR / "zepto_books.db"


def run_queries():
    conn = sqlite3.connect(DB_FILE)

    queries = {
        "Query 1 - Books with rating 5": """
            SELECT title, price_gbp, rating
            FROM books
            WHERE rating = 5;
        """,

        "Query 2 - Top 10 most expensive books": """
            SELECT title, price_gbp
            FROM books
            ORDER BY price_gbp DESC
            LIMIT 10;
        """,

        "Query 3 - Distinct ratings": """
            SELECT DISTINCT rating
            FROM books
            ORDER BY rating;
        """,

        "Query 4 - Books priced between £20 and £30": """
            SELECT title, price_gbp, rating
            FROM books
            WHERE price_gbp BETWEEN 20 AND 30
            ORDER BY price_gbp;
        """,

        "Query 5 - Books from selected categories": """
            SELECT title, category_id
            FROM books
            WHERE category_id IN (1, 2, 3);
        """,

        "Query 6 - Books with category names": """
            SELECT
                b.title,
                b.price_gbp,
                b.rating,
                c.category_name
            FROM books b
            JOIN categories c
                ON b.category_id = c.category_id
            ORDER BY b.price_gbp DESC
            LIMIT 15;
        """
    }

    output_file = BASE_DIR / "query_outputs.txt"

    with open(output_file, "w", encoding="utf-8") as f:

        for name, query in queries.items():

            print("\n" + "=" * 70)
            print(name)
            print("=" * 70)

            f.write("\n" + "=" * 70 + "\n")
            f.write(name + "\n")
            f.write("=" * 70 + "\n")

            # Save SQL query
            f.write("\nSQL:\n")
            f.write(query.strip() + "\n")

            cursor = conn.execute(query)

            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()

            print(columns)
            f.write("\nOUTPUT:\n")
            f.write(str(columns) + "\n")

            for row in rows:
                print(row)
                f.write(str(row) + "\n")

    conn.close()

    print("\nAll SQL queries executed successfully!")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    run_queries()