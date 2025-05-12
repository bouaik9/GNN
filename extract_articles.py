import undetected_chromedriver as uc
import json
import time
import csv

chrome_path = "chrome-linux64/chrome"

def scrape_and_extract_articles(url, depth, max_depth, articles, relations):
    if depth > max_depth:
        return None

    options = uc.ChromeOptions()
    options.binary_location = chrome_path
    driver = uc.Chrome(options=options)

    try:
        driver.get(url)
        print(f"Scraping URL: {url}")
        print("Page title is: ", driver.title)

        # Extract attributes directly using the driver
        title = driver.find_element("css selector", "h1.heading-title").text
        authors = [author.text for author in driver.find_elements("css selector", "span.authors-list-item")]
        abstract = driver.find_element("css selector", "div.abstract").text

        # Add the current article to the articles list
        article_id = len(articles) + 1
        articles.append({
            'id': article_id,
            'title': title,
            'authors': "; ".join(authors),
            'abstract': abstract,
            'url': url
        })

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

        # Recursively scrape cited articles
        for cited_article in cited_articles:
            time.sleep(2)  # Add delay to avoid being blocked
            cited_article_data = scrape_and_extract_articles(cited_article['link'], depth + 1, max_depth, articles, relations)
            if cited_article_data:
                # Add a relation between the current article and the cited article
                relations.append({
                    'source_id': article_id,
                    'target_id': len(articles)  # The cited article ID will be the next in the list
                })

        # Return extracted data
        return {
            'title': title,
            'authors': authors,
            'abstract': abstract,
            'cited_articles': cited_articles
        }

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

    finally:
        driver.quit()


if __name__ == "__main__":
    # Example URL (replace with the actual URL you want to scrape)
    url = "https://pubmed.ncbi.nlm.nih.gov/30519881/"
    max_depth = 2  # Set the maximum depth for recursive scraping

    # Lists to store articles and relations
    articles = []
    relations = []

    # Scrape and save data
    scrape_and_extract_articles(url, depth=1, max_depth=max_depth, articles=articles, relations=relations)

    # Write articles to CSV
    with open("articles.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "title", "authors", "abstract", "url"])
        writer.writeheader()
        writer.writerows(articles)

    # Write relations to CSV
    with open("relations.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_id", "target_id"])
        writer.writeheader()
        writer.writerows(relations)

    print("Data saved to articles.csv and relations.csv")
