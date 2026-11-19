import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_full_platform():
    print("=================================================================")
    print("RUNNING COMPREHENSIVE ARCHITECTURAL VERIFICATION SUITE (V2.0)")
    print("=================================================================")

    # 1. Test Gateway Health (/v1/health)
    print("\n[TEST 1] Testing Gateway Telemetry (GET /v1/health)...")
    r = requests.get(f"{BASE_URL}/v1/health")
    assert r.status_code == 200, f"Health failed: {r.status_code}"
    h_data = r.json()
    print("Status:", h_data.get("status"))
    print("Gateway:", h_data.get("gateway"))
    print("Protocol:", h_data.get("protocol"))
    print("Latency:", h_data.get("latency_ms"), "ms")
    assert h_data.get("status") == "healthy"
    print("[PASS] Gateway Telemetry Verified")

    # 2. Test Models Catalog (/v1/models)
    print("\n[TEST 2] Testing OpenAI Models Catalog (GET /v1/models)...")
    r = requests.get(f"{BASE_URL}/v1/models")
    assert r.status_code == 200
    models_data = r.json()
    m_ids = [m["id"] for m in models_data.get("data", [])]
    print("Registered Models:", m_ids)
    assert "llama-3.3-70b-versatile" in m_ids
    assert "flux-1-dev" in m_ids
    assert "custom-loras" in m_ids
    print("[PASS] Models Catalog Verified")

    # 3. Test OpenAI Chat Completions (/v1/chat/completions)
    print("\n[TEST 3] Testing OpenAI Chat Completions (POST /v1/chat/completions)...")
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": "What is the recommended seam sealing temperature for 3L technical shells?"}],
        "temperature": 0.2
    }
    r = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
    assert r.status_code == 200
    comp_data = r.json()
    print("Response ID:", comp_data.get("id"))
    print("Model:", comp_data.get("model"))
    print("Reply:", comp_data["choices"][0]["message"]["content"][:100], "...")
    print("[PASS] OpenAI Chat Completions Verified")

    # 4. Test Module 1: Tech-Pack & Mermaid Flowchart (/api/techpack/generate)
    print("\n[TEST 4] Testing Module 1: Tech-Pack & Dynamic Mermaid Flowchart...")
    payload = {"prompt": "Traditional 8-panel wool cricket cap with embroidered crest and curved stiff brim"}
    r = requests.post(f"{BASE_URL}/api/techpack/generate", json=payload)
    assert r.status_code == 200
    tp_data = r.json()
    print("Product Name:", str(tp_data["specs"].get("product_name")).encode('ascii', 'replace').decode('ascii'))
    print("Category:", str(tp_data["specs"].get("category")).encode('ascii', 'replace').decode('ascii'))
    mw = str(tp_data.get("mermaid_workflow")).encode('ascii', 'replace').decode('ascii')
    print("Mermaid Workflow:\n", mw)
    print("CAD Image URL:", tp_data.get("cad_image_url"))
    print("PDF URL:", tp_data.get("pdf_url"))
    assert "graph" in tp_data.get("mermaid_workflow", "")
    assert tp_data.get("pdf_url").endswith(".pdf")
    print("[PASS] Module 1 Tech-Pack & Mermaid Generation Verified")

    # 5. Test Module 2: Social Connectors (/api/social/sync-youtube & /api/social/post-facebook)
    print("\n[TEST 5] Testing Module 2: Social Middleware...")
    r_yt = requests.get(f"{BASE_URL}/api/social/sync-youtube")
    assert r_yt.status_code == 200
    print("YouTube Sync items:", r_yt.json().get("count"))
    
    r_fb = requests.post(f"{BASE_URL}/api/social/post-facebook", json={"message": "Broadcasting production readiness."})
    assert r_fb.status_code == 200
    print("Facebook Post:", r_fb.json().get("message"))

    r_reply = requests.post(f"{BASE_URL}/api/social/auto-reply", json={"platform": "meta", "comment": "What is the MOQ for custom colorways?", "author": "Elena"})
    assert r_reply.status_code == 200
    print("Auto-Reply:", r_reply.json().get("reply"))
    print("[PASS] Module 2 Social Middleware Verified")

    # 6. Test Module 3: MLOps Continuous Fine-Tuning Pipeline
    print("\n[TEST 6] Testing Module 3: MLOps Continuous Fine-Tuning Pipeline...")
    r_ad = requests.get(f"{BASE_URL}/api/mlops/adapters")
    assert r_ad.status_code == 200
    print("Initial Adapters Count:", len(r_ad.json().get("adapters", [])))

    r_start = requests.post(f"{BASE_URL}/api/mlops/start-fine-tuning", json={
        "base_model": "Meta Llama 3.3 (70B)",
        "lora_rank": 32,
        "epochs": 3,
        "learning_rate": "2e-4"
    })
    assert r_start.status_code == 200
    print("Training Job Started:", r_start.json().get("job_id"))

    time.sleep(2)
    r_st = requests.get(f"{BASE_URL}/api/mlops/training-status")
    assert r_st.status_code == 200
    st_data = r_st.json()
    print(f"Telemetry: Progress={st_data.get('progress')}% | Epoch={st_data.get('current_epoch')} | Step={st_data.get('current_step')} | Loss={st_data.get('current_loss')}")
    assert st_data.get("current_loss") <= 2.45
    print("[PASS] Module 3 MLOps Pipeline Verified")

    # 7. Test Frontend (index.html structure)
    print("\n[TEST 7] Testing Frontend SPA HTML Structure...")
    r_html = requests.get(f"{BASE_URL}/")
    assert r_html.status_code == 200
    html = r_html.text
    assert "left-sidebar" in html, "Sidebar missing"
    assert "presentation-pane" in html, "CIO Presentation pane missing"
    assert "mermaid.initialize" in html, "Mermaid missing"
    assert "Dynamic Factory Assembly Sequence (Mermaid.js)" in html, "Mermaid card missing"
    assert "Autonomous MLOps Continuous Fine-Tuning Pipeline" in html, "MLOps UI missing"
    assert "Enterprise OpenAI-Compatible SDK Gateway (/v1)" in html, "Gateway UI missing"
    print("[PASS] Frontend 3-Column Architecture & All Modules Verified")

    print("\n=================================================================")
    print("ALL 7 ARCHITECTURAL VERIFICATION SUITES PASSED WITH 100% SUCCESS!")
    print("=================================================================")

if __name__ == "__main__":
    test_full_platform()
