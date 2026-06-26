import requests

def submit_sign_detection(text, node_id, token, map_type="map_z"):
    
    url = "https://hackathon2025-dev.fpt.edu.vn/api/sign-submissions/submit/"
    
    payload = {
        "text": text,
        "node_id": node_id,
        "token": token,
        "map_type": map_type
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 201:
            print("✅ Submit thành công!")
            return response.json()
        else:
            print("❌ Submit thất bại.")
            print(f"Lỗi: {response.status_code}")
            print(response.json())
            return None
    except Exception as e:
        print(f"❌ Lỗi kết nối: {e}")
        return None
