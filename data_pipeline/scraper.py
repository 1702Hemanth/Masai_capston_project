import requests
from bs4 import BeautifulSoup
import pandas as pd


BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"

RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

GBP_TO_INR = 105.50


def scrape_books():
    """
    Scrape books from the first 5 pages of books.toscrape.com.
    """

    books = []

    for page in range(1, 6):

        url = BASE_URL.format(page)

        response = requests.get(url, timeout=10)

        # Check whether the request was successful
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
            availability = book.select_one(".availability").get_text(
                " ",
                strip=True
            )

            # Category
            category = get_category(book)

            books.append({
                "title": title,
                "price": price,
                "star_rating": star_rating,
                "availability": availability,
                "category": category
            })

    return pd.DataFrame(books)


def get_category(book):
    """
    Extract category information from the book listing.

    The category is obtained from the book's detail page.
    """

    book_link = book.h3.a["href"]

    # Convert relative URL into a usable URL
    if book_link.startswith("../"):
        book_link = book_link.replace("../", "")

    detail_url = "https://books.toscrape.com/catalogue/" + book_link

    response = requests.get(detail_url, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    breadcrumb = soup.select("ul.breadcrumb li a")

    if len(breadcrumb) >= 3:
        return breadcrumb[2].get_text(strip=True)

    return "Unknown"


def clean_books(df):
    """
    Clean scraped book data and create properly typed columns.
    """

    # Convert price from £51.77 → 51.77
    df["price_gbp"] = (
        df["price"]
        .str.replace("£", "", regex=False)
        .str.replace("Â", "", regex=False)
        .astype(float)
    )
# Convert to numeric, coercing errors to NaN
    df["price_gdp"]=pd.to_numeric(
        df["price_gbp"],
        errors='coerce')  

    # Convert One/Two/Three/Four/Five → 1/2/3/4/5
    df["rating"] = df["star_rating"].map(RATING_MAP)

    # Convert availability text to boolean
    df["in_stock"] = df["availability"].str.contains(
        "In stock",
        case=False,
        na=False
    )

    # Handle unexpected rating values
    if df["rating"].isna().any():
        median_rating = df["rating"].median()
        df["rating"] = df["rating"].fillna(median_rating)

    # Convert rating to integer
    df["rating"] = df["rating"].round().astype(int)

    # Handle unexpected price values
    if df["price_gbp"].isna().any():
        median_price = df["price_gbp"].median()
        df["price_gbp"] = df["price_gbp"].fillna(median_price)

    # Required fixed conversion
    df["price_inr"] = df["price_gbp"] * GBP_TO_INR

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

    # Save intermediate cleaned data
    df.to_csv("data_pipeline/cleaned_books.csv", index=False)

    print("\nCleaned dataset saved to:")
    print("data_pipeline/cleaned_books.csv")


if __name__ == "__main__":
    main()