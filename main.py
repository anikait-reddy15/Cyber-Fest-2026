from src.database import init_db, save_app
from src.scraper import get_live_training_data
from src.analyzer import RiskModel, analyze_app_ml

def main():
    # 1. Setup DB
    init_db()
    
    # 2. Get Data & Train
    print("\n--- 📡 PHASE 1: LIVE TRAINING ---")
    df_train = get_live_training_data()
    
    model = RiskModel()
    model.train(df_train) # Calls the Qwen training logic
    
    # 3. Test Scan
    print("\n--- 🕵️ PHASE 2: AUTO-SCANNING TEST APPS ---")
    test_app = {"name": "InstaView Hack", "description": "Unlock profiles.", "permissions": "admin", "publisher": "Hacker"}
    
    score, reason = analyze_app_ml(test_app)
    test_app['risk_score'] = score
    test_app['risk_reasons'] = reason
    save_app(test_app)
    
    print(f"Result: {score}% Risk")

if __name__ == "__main__":
    main()