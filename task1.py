import requests
from bs4 import BeautifulSoup
import csv

# URL of the page to scrape
url = "https://hsls.libguides.com/health-data-sources/data-sets"

# Send a request to fetch the webpage
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    print("Page fetched successfully!")
    
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all dataset links on the page
    dataset_links = soup.find_all('a', href=True)

    # Open a CSV file to save the data
    with open('health_datasets.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset Name", "Link"])  # Header

        # Extract and write data
        for link in dataset_links:
            title = link.text.strip()
            href = link['href']
            if title and href.startswith("http"):  # Ensure it's a valid external link
                writer.writerow([title, href])
                print(f"Dataset: {title}")
                print(f"Link: {href}")
                print("-" * 50)

    print("✅ Data successfully saved to health_datasets.csv")

else:
    print("❌ Failed to retrieve the page")


