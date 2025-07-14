from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time

# === Configuration ===
MAX_TWEETS = 600

# List of Twitter handles (without '@')
accounts = [
    #"OPECSecretariat",
    #"IEA",
    #"KremlinRussia_E",
    #"mfa_russia",
    #"WhiteHouse",
    #"POTUS",
    #"IsraeliPM",
    #"IDF",
    #"IRIran_Military",
    #"JavierBlas",
    #"OilSheppard",
    #"DanielYergin",
    #"SStapczynski",
    #"zerohedge",
    #"chigrl",
    #"BurggrabenH",
    #"sentdefender",
    #"IntelDoge",
    "TrumpDailyPosts",
    #"Yemenimilitary",
    #"OSE_Yemen",
]

def setup_driver():
    options = Options()
    options.add_argument("--headless=new")  # Better headless mode
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-gpu")
    options.add_argument("window-size=1200x900")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.90 Safari/537.36")
    
    driver = webdriver.Chrome(options=options)
    driver.set_window_size(1200, 900)
    return driver

def scrape_tweets(driver, username, max_tweets=100):
    #url = f"https://nitter.poast.org/{username}"
    url = f"https://nitter.tiekoetter.com/{username}"
    driver.get(url)
    time.sleep(2)

    collected = []
    seen = set()

    while len(collected) < max_tweets:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        try:
            load_more = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//div[@class='show-more']/a"))
            )
            driver.execute_script("arguments[0].click();", load_more)
            print("🔁 Clicked 'Load more'")
            time.sleep(2)
        except Exception:
            print("❌ 'Load more' button not found or no longer clickable.")
            break

        soup = BeautifulSoup(driver.page_source, "html.parser")
        tweets = soup.find_all("div", class_="timeline-item")

        for tweet in tweets:
            text_tag = tweet.find("div", class_="tweet-content")
            time_tag = tweet.find("span", class_="tweet-date")
            if not text_tag or not time_tag:
                continue

            content = text_tag.text.strip()
            timestamp_a = time_tag.find("a")
            timestamp = timestamp_a.get("title", "N/A") if timestamp_a else "N/A"

            tweet_id = f"{timestamp}-{content}"
            if tweet_id not in seen:
                collected.append([timestamp, content])
                seen.add(tweet_id)

            if len(collected) >= max_tweets:
                break

    return collected

def save_to_excel(tweets, username):
    df = pd.DataFrame(tweets, columns=["Timestamp", "Tweet"])
    filename = f"{username}_tweets.xlsx"
    df.to_excel(filename, index=False)
    print(f"\n✅ Saved {len(tweets)} tweets to {filename}")

if __name__ == "__main__":
    driver = setup_driver()
    try:
        for username in accounts:
            print(f"\n🌐 Scraping @{username}...")
            tweets = scrape_tweets(driver, username, max_tweets=MAX_TWEETS)
            if tweets:
                save_to_excel(tweets, username)
    finally:
        driver.quit()
        
print("Downloading complete. All tweets saved to Excel files.")