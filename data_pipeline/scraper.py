import requests
import time
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path


BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

GBP_TO_INR = 105.50

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "cleaned_books.csv"


def scrape_books():
    """
    Scrape books from the first 5 pages of books.toscrape.com.
    """

    books = []

    for page in range(1, 6):

        url = BASE_URL.format(page)

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        book_items = soup.select("article.product_pod")

        for book in book_items:

            # Title
            title = book.h3.a["title"]

            # Price
            price = book.select_one(".price_color").get_text(strip=True)

            # Rating
            rating_element = book.select_one(".star-rating")
            star_rating = rating_element.get("class")[1]

            # Availability
            availability = book.select_one(
                ".availability"
            ).get_text(" ", strip=True)

            # Category
            category = get_category(book)

            books.append({
                "title": title,
                "price": price,
                "star_rating": star_rating,
                "availability": availability,
                "category": category
            })
            time.sleep(0.2)  # Be polite and avoid overwhelming the server

    return pd.DataFrame(books)

def get_category(book):
    """
    Extract the actual category from the book detail page.
    Uses a persistent session and retries temporary network failures.
    """

    book_link = book.h3.a["href"]

    if book_link.startswith("../"):
        book_link = book_link.replace("../", "", 1)

    detail_url = "https://books.toscrape.com/catalogue/" + book_link

    # Valid categories from Books to Scrape
    VALID_CATEGORIES = {
        "Art", "Biography", "Business", "Childrens",
        "Christian", "Contemporary", "Crime", "Fantasy",
        "Fiction", "Food and Drink", "Health",
        "Historical Fiction", "History", "Horror", "Humor",
        "Music", "Mystery", "New Adult", "Nonfiction",
        "Paranormal", "Parenting", "Philosophy", "Poetry",
        "Politics", "Psychology", "Religion", "Romance",
        "Science", "Science Fiction", "Self Help",
        "Sequential Art", "Short Stories", "Spirituality",
        "Sports and Games", "Thriller", "Travel", "Young Adult"
    }

    # Use a persistent session
    session = requests.Session()

    for attempt in range(5):

        try:
            response = session.get(
                detail_url,
                timeout=(10, 60)
            )

            response.raise_for_status()

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            breadcrumb_items = soup.select(
                "ul.breadcrumb > li"
            )

            # Search the breadcrumb for a VALID category
            for item in breadcrumb_items:

                text = item.get_text(
                    strip=True
                )

                if text in VALID_CATEGORIES:
                    return text

            return "Unknown"

        except requests.RequestException as error:

            print(
                f"Category request failed "
                f"(attempt {attempt + 1}/5): "
                f"{detail_url}"
            )

    return "Unknown"

def clean_books(df):
    """
    Clean scraped book data and create properly typed columns.
    Rows with unparsable categories will be dropped.
    """

    before=len(df)
    df=df[df["category"]!="Unknown"].copy()
    dropped=before-len(df)

    print(f"Dropped {dropped} rows with unknown categories.")
    print(f"Remaining rows: {len(df)}")

    # -----------------------------
    # Clean price
    # -----------------------------

    df["price_gbp"] = (
        df["price"]
        .astype(str)
        .str.replace("£", "", regex=False)
        .str.replace("Â", "", regex=False)
        .str.strip()
    )

    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce"
    )

    # Median imputation for invalid prices
    if df["price_gbp"].isna().any():

        median_price = df["price_gbp"].median()

        df["price_gbp"] = df["price_gbp"].fillna(median_price)

    # -----------------------------
    # Clean rating
    # -----------------------------

    df["rating"] = df["star_rating"].map(RATING_MAP)

    # Median imputation for unexpected ratings
    if df["rating"].isna().any():

        median_rating = df["rating"].median()

        df["rating"] = df["rating"].fillna(median_rating)

    df["rating"] = (
        df["rating"]
        .round()
        .astype(int)
    )

    # -----------------------------
    # Clean availability
    # -----------------------------

    df["in_stock"] = df["availability"].str.contains(
        "In stock",
        case=False,
        na=False
    )

    # -----------------------------
    # Fixed GBP → INR conversion
    # -----------------------------

    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    )

    return df


def main():

    print("Starting book scraping...")

    df = scrape_books()

    print(f"Books scraped: {len(df)}")

    df = clean_books(df)

    print("\nCleaned data:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nNumber of categories:")
    print(df["category"].nunique())

    print("\nCategories:")
    print(df["category"].unique())

    # Save cleaned dataset
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nCleaned dataset saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()