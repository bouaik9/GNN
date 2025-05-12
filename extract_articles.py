import undetected_chromedriver as uc

chrome_path = "chrome-linux64/chrome"

def scrape_and_extract_articles(url):
    options = uc.ChromeOptions()
    options.binary_location = chrome_path
    driver = uc.Chrome(options=options)

    try:
        driver.get(url)
        print("Page title is: ", driver.title)

        # Extract attributes directly using the driver
        title = driver.find_element("css selector", "h1.heading-title").text
        authors = [author.text for author in driver.find_elements("css selector", "span.authors-list-item")]
        abstract = driver.find_element("css selector", "div.abstract").text

        cited_articles = []
        try:
            cited_section = driver.find_element("css selector", "div.citedby-articles")
            cited_items = cited_section.find_elements("css selector", "li.full-docsum")
            for item in cited_items:
                cited_title = item.find_element("css selector", "a.docsum-title").text
                cited_link = item.find_element("css selector", "a.docsum-title").get_attribute("href")
                cited_authors = item.find_element("css selector", "span.docsum-authors.full-authors").text
                cited_articles.append({
                    'title': cited_title,
                    'link': cited_link,
                    'authors': cited_authors
                })
        except Exception:
            print("No cited articles section found.")

        # Print extracted data
        print(f"Title: {title}")
        print(f"Authors: {', '.join(authors)}")
        print(f"Abstract: {abstract}")
        print("Cited Articles:")
        for article in cited_articles:
            print(f"  - Title: {article['title']}")
            print(f"    Link: {article['link']}")
            print(f"    Authors: {article['authors']}")

    finally:
        driver.quit()


if __name__ == "__main__":
    # Example URL (replace with the actual URL you want to scrape)
    url = "https://pubmed.ncbi.nlm.nih.gov/30519881/"
    scrape_and_extract_articles(url)
