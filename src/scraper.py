import pandas as pd
import random
from google_play_scraper import search

def scrape_live_legit_apps(count=40): # Increased count
    print(f"🌐 Connecting to Google Play Store to fetch top {count} apps...")
    
    search_terms = ["social", "communication", "tools", "productivity", "finance", "games"]
    results = []
    
    for term in search_terms:
        try:
            scraped = search(term, lang='en', country='us', n_hits=10)
            results.extend(scraped)
            if len(results) >= count:
                break
        except Exception as e:
            print(f"Warning: {e}")
            
    legit_apps = []
    for item in results[:count]:
        legit_apps.append({
            "name": item['title'],
            "desc": item.get('description', item.get('summary', ''))[:200], # Truncate for speed
            "perms": "internet access, storage",
            "publisher": item['developer'],
            "label": 0 # Real
        })
        
    return legit_apps

def generate_synthetic_fakes(legit_apps):
    print(f"🧪 Generating aggressive fake apps...")
    fake_apps = []
    
    # Aggressive Keywords
    prefixes = ["Free", "Unlimited", "Pro", "Cracked", "Mod", "Hack", "Generator"]
    suffixes = ["Gold", "Plus", "Premium", "2026", "Glitch", "Unlocker"]
    
    bad_descriptions = [
        "Get unlimited coins and gems for free.",
        "Unlock private profiles and view hidden photos.",
        "Generate free money to your account instantly.",
        "Bypass verification and get admin access.",
        "Cheat tool for 100% winning rate.",
        "Download videos from private accounts.",
        "Free premium features unlocked no root."
    ]
    
    risky_perms = " READ_SMS, SEND_SMS, SYSTEM_ALERT_WINDOW, INSTALL_PACKAGES, READ_CONTACTS"
    
    for app in legit_apps:
        # Create a fake version of a real app
        fake_name = f"{random.choice(prefixes)} {app['name']} {random.choice(suffixes)}"
        
        # 50% chance to replace description entirely with a scam description
        if random.random() > 0.5:
            fake_desc = random.choice(bad_descriptions)
        else:
            fake_desc = f"Unlocking {app['name']} features. {random.choice(bad_descriptions)}"

        fake_apps.append({
            "name": fake_name,
            "desc": fake_desc,
            "perms": app['perms'] + risky_perms,
            "publisher": "Unknown Dev", # Generic bad publisher
            "label": 1 # Fake
        })
        
    return fake_apps

def get_live_training_data():
    legit_data = scrape_live_legit_apps(count=40)
    fake_data = generate_synthetic_fakes(legit_data)
    
    full_data = legit_data + fake_data
    random.shuffle(full_data)
    
    return pd.DataFrame(full_data)