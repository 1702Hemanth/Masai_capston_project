import sqlite3
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_FILE = BASE_DIR / "zepto_books.db"


def main():

    conn = sqlite3.connect(DB_FILE)

    # --------------------------------------------------
    # 1. Read SQL Query Results using pd.read_sql()
    # --------------------------------------------------

    query_rating = """
        SELECT title, price_gbp, rating
        FROM books
        WHERE rating = 5
        ORDER BY price_gbp DESC
        LIMIT 10;
    """

    rating_df = pd.read_sql(query_rating, conn)

    print("\n=== pd.read_sql() - Rating Query ===")
    print(rating_df.to_string(index=False))

    # --------------------------------------------------
    # 2. Read JOIN result using pd.read_sql()
    # --------------------------------------------------

    join_query = """
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

    sql_join_df = pd.read_sql(join_query, conn)

    print("\n=== pd.read_sql() - JOIN Result ===")
    print(sql_join_df.to_string(index=False))

    # --------------------------------------------------
    # 3. Read source tables into pandas
    # --------------------------------------------------

    books_df = pd.read_sql(
        "SELECT * FROM books",
        conn
    )

    categories_df = pd.read_sql(
        "SELECT * FROM categories",
        conn
    )

    # --------------------------------------------------
    # 4. Reproduce the JOIN using pd.merge()
    # --------------------------------------------------

    merge_df = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )

    merge_df = merge_df[
        [
            "title",
            "price_gbp",
            "rating",
            "category_name"
        ]
    ]

    merge_df = (
        merge_df
        .sort_values("price_gbp", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    sql_join_df = (
        sql_join_df
        .sort_values("price_gbp", ascending=False)
        .reset_index(drop=True)
    )

    print("\n=== pd.merge() - JOIN Result ===")
    print(merge_df.to_string(index=False))

    # --------------------------------------------------
    # 5. Compare SQL JOIN and pandas JOIN
    # --------------------------------------------------

    equivalent = sql_join_df.equals(merge_df)

    print("\n=== Comparison ===")
    print(f"SQL JOIN and pd.merge() equivalent: {equivalent}")

    # Save results
    output_file = BASE_DIR / "pandas_outputs.txt"

    with open(output_file, "w", encoding="utf-8") as f:

        f.write("=== pd.read_sql() - Rating Query ===\n")
        f.write(rating_df.to_string(index=False))

        f.write("\n\n=== pd.read_sql() - JOIN Result ===\n")
        f.write(sql_join_df.to_string(index=False))

        f.write("\n\n=== pd.merge() - JOIN Result ===\n")
        f.write(merge_df.to_string(index=False))

        f.write("\n\n=== Comparison ===\n")
        f.write(
            f"SQL JOIN and pd.merge() equivalent: {equivalent}\n"
        )

    conn.close()

    print(f"\nPandas output saved to: {output_file}")


if __name__ == "__main__":
    main()