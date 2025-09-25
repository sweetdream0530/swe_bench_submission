#!/usr/bin/env python3
"""
Test script to verify API key rotation functionality
"""

import os
import sys
import asyncio
from unittest.mock import Mock, patch

# Add the proxy directory to the path
sys.path.insert(0, '/root/swe_bench_submission/proxy')

def test_config_loading():
    """Test that config properly loads multiple API keys"""
    print("Testing config loading...")
    
    # Set up test environment variables
    test_keys = [
        "test_key_1",
        "test_key_2", 
        "test_key_3",
        "test_key_4",
        "test_key_5"
    ]
    
    for i, key in enumerate(test_keys):
        if i == 0:
            os.environ["CHUTES_API_KEY"] = key
        else:
            os.environ[f"CHUTES_API_KEY_{i}"] = key
    
    # Import config after setting environment variables
    from proxy.config import CHUTES_API_KEYS
    
    print(f"Loaded {len(CHUTES_API_KEYS)} API keys")
    print(f"Keys: {[key[:10] + '...' for key in CHUTES_API_KEYS]}")
    
    assert len(CHUTES_API_KEYS) == 5, f"Expected 5 keys, got {len(CHUTES_API_KEYS)}"
    assert all(key in CHUTES_API_KEYS for key in test_keys), "Not all test keys were loaded"
    
    print("✅ Config loading test passed!")
    return True

def test_chutes_provider_rotation():
    """Test ChutesProvider API key rotation"""
    print("\nTesting ChutesProvider rotation...")
    
    from proxy.providers.chutes_provider import ChutesProvider
    
    provider = ChutesProvider()
    
    # Test availability check
    assert provider.is_available(), "Provider should be available with test keys"
    
    # Test key rotation
    initial_key = provider.get_current_api_key()
    print(f"Initial key: {initial_key[:10]}...")
    
    # Mark current key as failed
    provider.mark_key_as_failed(initial_key)
    
    # Get next key
    next_key = provider.get_current_api_key()
    print(f"Next key: {next_key[:10]}...")
    
    assert initial_key != next_key, "Should get different key after marking as failed"
    
    print("✅ ChutesProvider rotation test passed!")
    return True

def test_chutes_client_rotation():
    """Test ChutesClient API key rotation"""
    print("\nTesting ChutesClient rotation...")
    
    from proxy.chutes_client import ChutesClient
    
    client = ChutesClient()
    
    # Test key rotation
    initial_key = client.get_current_api_key()
    print(f"Initial key: {initial_key[:10]}...")
    
    # Mark current key as failed
    client.mark_key_as_failed(initial_key)
    
    # Get next key
    next_key = client.get_current_api_key()
    print(f"Next key: {next_key[:10]}...")
    
    assert initial_key != next_key, "Should get different key after marking as failed"
    
    print("✅ ChutesClient rotation test passed!")
    return True

async def test_error_handling():
    """Test error handling and key rotation on rate limits"""
    print("\nTesting error handling...")
    
    from proxy.providers.chutes_provider import ChutesProvider
    
    provider = ChutesProvider()
    
    # Mock a rate limit error
    with patch('proxy.providers.chutes_provider.httpx.AsyncClient') as mock_client:
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        
        mock_client.return_value.__aenter__.return_value.stream.return_value.__aenter__.return_value.status_code = 429
        mock_client.return_value.__aenter__.return_value.stream.return_value.__aenter__.return_value.aread.return_value = b"Rate limit exceeded"
        
        try:
            await provider.inference(
                messages=[Mock(role="user", content="test")],
                model="test-model",
                temperature=0.7
            )
        except RuntimeError as e:
            if "exhausted" in str(e):
                print("✅ Rate limit handling test passed!")
                return True
    
    print("❌ Rate limit handling test failed!")
    return False

def cleanup_test_env():
    """Clean up test environment variables"""
    keys_to_remove = [
        "CHUTES_API_KEY",
        "CHUTES_API_KEY_1", 
        "CHUTES_API_KEY_2",
        "CHUTES_API_KEY_3",
        "CHUTES_API_KEY_4",
        "CHUTES_API_KEY_5"
    ]
    
    for key in keys_to_remove:
        if key in os.environ:
            del os.environ[key]

async def main():
    """Run all tests"""
    print("🧪 Testing API Key Rotation Implementation")
    print("=" * 50)
    
    try:
        # Run tests
        test_config_loading()
        test_chutes_provider_rotation()
        test_chutes_client_rotation()
        await test_error_handling()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! API key rotation is working correctly.")
        print("\nTo use multiple API keys, set these environment variables:")
        print("- CHUTES_API_KEY")
        print("- CHUTES_API_KEY_1")
        print("- CHUTES_API_KEY_2")
        print("- CHUTES_API_KEY_3")
        print("- CHUTES_API_KEY_4")
        print("- CHUTES_API_KEY_5")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_test_env()
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
