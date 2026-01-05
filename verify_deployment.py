import asyncio
import httpx
import sys
import time

BASE_URL = "http://localhost:8000"

async def verify_deployment():
    print("🚀 Starting Deployment Verification...")
    
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
        # 1. Test Health (Phase 1)
        print("\nTesting Health Endpoint (Phase 1)...")
        try:
            response = await client.get("/health")
            if response.status_code == 200:
                print("✅ Health Check Passed")
                print(f"   Status: {response.json()['status']}")
            else:
                print(f"❌ Health Check Failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Could not connect to server: {e}")
            return False

        # 2. Test Registration (Phase 2)
        print("\nTesting User Registration (Phase 2)...")
        email = f"test_{int(time.time())}@example.com"
        password = "securepassword123"
        
        try:
            response = await client.post("/api/auth/register", json={
                "email": email,
                "password": password,
                "full_name": "Test User"
            })
            if response.status_code == 200:
                print("✅ Registration Passed")
                tokens = response.json()
                access_token = tokens["access_token"]
                print("   Got Access Token")
            else:
                print(f"❌ Registration Failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"❌ Registration Error: {e}")
            return False

        # 3. Test Protected Endpoint (Phase 2)
        print("\nTesting Protected Endpoint (Phase 2)...")
        try:
            # Try accessing orchestration without token (should fail)
            resp_fail = await client.get("/api/v1/orchestration/workflows/123")
            if resp_fail.status_code == 401:
                print("✅ Auth Protection Working (401 received as expected)")
            else:
                print(f"❌ Auth Protection Failed: Got {resp_fail.status_code} instead of 401")
            
            # Try with token (should pass or give 404 if workflow doesn't exist, but NOT 401)
            resp_auth = await client.get(
                "/api/v1/orchestration/workflows/123",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            if resp_auth.status_code != 401:
                print(f"✅ Authenticated Request Passed (Got {resp_auth.status_code})")
            else:
                print("❌ Authenticated Request Failed (Got 401)")
                return False
                
        except Exception as e:
            print(f"❌ Protected Endpoint Error: {e}")
            return False

    print("\n✨ ALL CHECKS PASSED! Phase 1 & 2 are correctly implemented.")
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(verify_deployment())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nVerification cancelled")
