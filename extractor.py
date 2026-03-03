import os
import json
from androguard.core.bytecodes.apk import APK

def extract_metadata(apk_path):
    try:
        app = APK(apk_path)
        return {
            "permissions": app.get_permissions(),
            "activities": app.get_activities(),
            "services": app.get_services()
        }
    except Exception:
        return None

def process_folder(folder_path="data/fake_apps_150/"):
    dataset = []
    seen_metadata = set()
    for filename in os.listdir(folder_path):
        if filename.endswith(".apk"):
            apk_path = os.path.join(folder_path, filename)
            data = extract_metadata(apk_path)
            if data:
                data_str = json.dumps(data, sort_keys=True)
                if data_str not in seen_metadata:
                    seen_metadata.add(data_str)
                    dataset.append(data)
    return dataset