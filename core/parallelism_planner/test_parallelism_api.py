#!/usr/bin/env python3
"""
Test script for Parallelism Strategy Advisor API

Run this after starting the API server to verify all endpoints work correctly.

Usage:
    python test_parallelism_api.py
"""

import requests
import json
import sys
from typing import Dict, Any

# Configuration
API_URL = "http://localhost:8000"
TEST_MODEL = "meta-llama/Llama-3.1-70B"
MOCK_TOPOLOGY = "h100"
MOCK_GPUS = 8


def print_test_header(test_name: str):
    """Print a formatted test header."""
    print("\n" + "=" * 80)
    print(f"TEST: {test_name}")
    print("=" * 80)


def print_response(response: requests.Response, show_full: bool = False):
    """Print formatted response."""
    if response.status_code == 200:
        print(f"✓ Status: {response.status_code} OK")
        result = response.json()
        if show_full:
            print(json.dumps(result, indent=2))
        else:
            # Print abbreviated response
            print(f"Response keys: {list(result.keys())}")
            print(f"Response preview: {str(result)[:200]}...")
    else:
        print(f"✗ Status: {response.status_code} ERROR")
        print(response.text)
        

def test_health():
    """Test health check endpoint."""
    print_test_header("Health Check")
    response = requests.get(f"{API_URL}/health")
    print_response(response, show_full=True)
    return response.status_code == 200


def test_root():
    """Test root endpoint."""
    print_test_header("Root Endpoint")
    response = requests.get(f"{API_URL}/")
    print_response(response, show_full=True)
    return response.status_code == 200


def test_presets():
    """Test presets endpoint."""
    print_test_header("List Model Presets")
    response = requests.get(f"{API_URL}/presets")
    print_response(response)
    return response.status_code == 200


def test_topology():
    """Test topology detection."""
    print_test_header("Hardware Topology")
    response = requests.get(f"{API_URL}/topology?mock={MOCK_TOPOLOGY}&mock_gpus={MOCK_GPUS}")
    print_response(response)
    return response.status_code == 200


def test_recommend():
    """Test parallelism recommendations."""
    print_test_header("Parallelism Recommendations")
    payload = {
        "model": TEST_MODEL,
        "batch_size": 1,
        "seq_length": 2048,
        "goal": "throughput",
        "training": True,
        "mock_topology": MOCK_TOPOLOGY,
        "mock_gpus": MOCK_GPUS
    }
    response = requests.post(f"{API_URL}/recommend", json=payload)
    print_response(response)
    return response.status_code == 200


def test_sharding():
    """Test sharding recommendations."""
    print_test_header("Sharding Strategy")
    payload = {
        "model": TEST_MODEL,
        "dp": 8,
        "memory": 80,
        "batch_size": 1,
        "seq_length": 2048
    }
    response = requests.post(f"{API_URL}/sharding", json=payload)
    print_response(response)
    return response.status_code == 200


def test_launch():
    """Test launch command generation."""
    print_test_header("Launch Commands")
    payload = {
        "nodes": 2,
        "gpus": 8,
        "tp": 2,
        "pp": 2,
        "dp": 4,
        "sharding": "zero3",
        "micro_batch": 1,
        "grad_accum": 16,
        "script": "train.py"
    }
    response = requests.post(f"{API_URL}/launch", json=payload)
    print_response(response)
    return response.status_code == 200


def test_pareto():
    """Test Pareto analysis."""
    print_test_header("Pareto Analysis")
    payload = {
        "model": TEST_MODEL,
        "gpu_cost": 4.0,
        "batch_size": 8,
        "seq_length": 4096,
        "training": True
    }
    response = requests.post(f"{API_URL}/pareto", json=payload)
    print_response(response)
    return response.status_code == 200


def test_analyze():
    """Test model analysis."""
    print_test_header("Model Analysis")
    payload = {
        "model": TEST_MODEL,
        "batch_size": 8,
        "seq_length": 2048
    }
    response = requests.post(f"{API_URL}/analyze", json=payload)
    print_response(response)
    return response.status_code == 200


def test_estimate():
    """Test training time estimation."""
    print_test_header("Training Time & Cost Estimation")
    payload = {
        "model": TEST_MODEL,
        "tokens": 1_000_000_000_000,
        "throughput": 100000,
        "gpus": 64,
        "gpu_cost": 4.0,
        "checkpoint_interval": 1_000_000_000
    }
    response = requests.post(f"{API_URL}/estimate", json=payload)
    print_response(response, show_full=True)
    return response.status_code == 200


def test_validate():
    """Test configuration validation."""
    print_test_header("Configuration Validation")
    payload = {
        "model": TEST_MODEL,
        "tp": 2,
        "pp": 2,
        "dp": 4,
        "memory": 80,
        "batch_size": 1,
        "seq_length": 2048
    }
    response = requests.post(f"{API_URL}/validate", json=payload)
    print_response(response)
    return response.status_code == 200


def test_batchsize():
    """Test batch size finder."""
    print_test_header("Maximum Batch Size")
    payload = {
        "model": TEST_MODEL,
        "memory": 80,
        "seq_length": 2048,
        "tp": 2,
        "pp": 2
    }
    response = requests.post(f"{API_URL}/batchsize", json=payload)
    print_response(response)
    return response.status_code == 200


def test_inference():
    """Test inference optimization."""
    print_test_header("Inference Optimization")
    payload = {
        "model": TEST_MODEL,
        "batch_size": 1,
        "seq_length": 2048,
        "max_output_length": 512,
        "latency_target_ms": 100,
        "mock_topology": MOCK_TOPOLOGY,
        "mock_gpus": 1
    }
    response = requests.post(f"{API_URL}/inference", json=payload)
    print_response(response)
    return response.status_code == 200


def test_troubleshoot():
    """Test troubleshooting."""
    print_test_header("Troubleshooting")
    payload = {}
    response = requests.post(f"{API_URL}/troubleshoot", json=payload)
    print_response(response)
    return response.status_code == 200


def test_memory():
    """Test memory breakdown."""
    print_test_header("Memory Breakdown")
    payload = {
        "model": TEST_MODEL,
        "batch_size": 8,
        "seq_length": 2048,
        "tp": 2,
        "pp": 2,
        "training": True
    }
    response = requests.post(f"{API_URL}/memory", json=payload)
    print_response(response)
    return response.status_code == 200


def test_nccl():
    """Test NCCL tuning."""
    print_test_header("NCCL Tuning")
    payload = {
        "num_nodes": 4,
        "gpus_per_node": 8,
        "network_type": "nvlink"
    }
    response = requests.post(f"{API_URL}/nccl", json=payload)
    print_response(response, show_full=True)
    return response.status_code == 200


def run_all_tests():
    """Run all test cases."""
    print("=" * 80)
    print("Parallelism Strategy Advisor API - Test Suite")
    print("=" * 80)
    print(f"API URL: {API_URL}")
    print(f"Test Model: {TEST_MODEL}")
    
    # Check if server is running
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code != 200:
            print("\n✗ ERROR: API server is not responding correctly")
            print("Please start the server with: python parallelism_planner_server.py")
            sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"\n✗ ERROR: Cannot connect to API server at {API_URL}")
        print(f"Error: {e}")
        print("Please start the server with: python parallelism_planner_server.py")
        sys.exit(1)
    
    # Define test suite
    tests = [
        ("Health Check", test_health),
        ("Root Endpoint", test_root),
        ("Model Presets", test_presets),
        ("Hardware Topology", test_topology),
        ("Parallelism Recommendations", test_recommend),
        ("Sharding Strategy", test_sharding),
        ("Launch Commands", test_launch),
        ("Pareto Analysis", test_pareto),
        ("Model Analysis", test_analyze),
        ("Training Estimation", test_estimate),
        ("Configuration Validation", test_validate),
        ("Batch Size Finder", test_batchsize),
        ("Inference Optimization", test_inference),
        ("Troubleshooting", test_troubleshoot),
        ("Memory Breakdown", test_memory),
        ("NCCL Tuning", test_nccl),
    ]
    
    # Run tests
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' raised an exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for test_name, passed_test in results:
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("=" * 80)
    print(f"Total: {passed}/{total} tests passed ({100*passed//total}%)")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

