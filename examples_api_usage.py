#!/usr/bin/env python3
"""
Examples of using the Parallelism Strategy Advisor API

This script demonstrates various use cases and workflows using the API client.
"""

from parallelism_api_client import ParallelismAPIClient, APIError
import json
import sys


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_result(result: dict, keys: list = None, max_length: int = 500):
    """Print formatted result."""
    if keys:
        filtered = {k: result.get(k) for k in keys if k in result}
        print(json.dumps(filtered, indent=2))
    else:
        output = json.dumps(result, indent=2)
        if len(output) > max_length:
            print(output[:max_length] + "...")
        else:
            print(output)


def example_1_basic_recommendations(client: ParallelismAPIClient):
    """Example 1: Get basic parallelism recommendations."""
    print_section("Example 1: Basic Parallelism Recommendations")
    
    print("Getting recommendations for Llama-3.1-70B training on 8 H100 GPUs...")
    
    try:
        result = client.recommend(
            model="meta-llama/Llama-3.1-70B",
            batch_size=1,
            seq_length=2048,
            goal="throughput",
            training=True,
            mock_topology="h100",
            mock_gpus=8
        )
        print("\n✓ Recommendations received")
        print_result(result)
    except APIError as e:
        print(f"\n✗ Error: {e}")


def example_2_multi_node_setup(client: ParallelismAPIClient):
    """Example 2: Generate launch commands for multi-node training."""
    print_section("Example 2: Multi-Node Training Setup")
    
    print("Generating launch commands for 2-node, 8-GPU per node training...")
    print("Configuration: TP=2, PP=2, DP=4, ZeRO-3")
    
    try:
        result = client.launch(
            nodes=2,
            gpus=8,
            tp=2,
            pp=2,
            dp=4,
            sharding="zero3",
            micro_batch=1,
            grad_accum=16,
            master_addr="node1.cluster.local",
            master_port=29500,
            script="train_llama.py"
        )
        
        print("\n✓ Launch commands generated")
        
        # Print torchrun command
        if "commands" in result and "torchrun" in result["commands"]:
            print("\nTorchrun command:")
            print(result["commands"]["torchrun"].get("command", "N/A"))
        
        # Print DeepSpeed command
        if "commands" in result and "deepspeed" in result["commands"]:
            print("\nDeepSpeed command:")
            print(result["commands"]["deepspeed"].get("command", "N/A"))
            
    except APIError as e:
        print(f"\n✗ Error: {e}")


def example_3_cost_analysis(client: ParallelismAPIClient):
    """Example 3: Training cost estimation."""
    print_section("Example 3: Training Cost Estimation")
    
    print("Estimating cost for training Llama-3.1-70B on 1T tokens...")
    print("Configuration: 64 GPUs, 100K tokens/sec throughput")
    
    try:
        result = client.estimate(
            model="meta-llama/Llama-3.1-70B",
            tokens=1_000_000_000_000,  # 1T tokens
            throughput=100000,  # 100K tokens/sec
            gpus=64,
            gpu_cost=4.0,  # $4/GPU/hour
            checkpoint_interval=1_000_000_000  # Checkpoint every 1B tokens
        )
        
        print("\n✓ Cost estimation complete")
        print(f"\nTraining Time: {result.get('training_time_days', 'N/A')} days")
        print(f"Total Cost: ${result.get('total_cost_usd', 'N/A'):,.2f}")
        print(f"Cost per GPU: ${result.get('cost_per_gpu_usd', 'N/A'):,.2f}")
        print(f"Checkpoints: {result.get('num_checkpoints', 'N/A')}")
        print(f"Storage Required: {result.get('total_checkpoint_storage_gb', 'N/A')} GB")
        
    except APIError as e:
        print(f"\n✗ Error: {e}")


def example_4_pareto_analysis(client: ParallelismAPIClient):
    """Example 4: Pareto frontier analysis."""
    print_section("Example 4: Cost vs Throughput Pareto Analysis")
    
    print("Analyzing cost-throughput tradeoffs for Llama-3.1-70B...")
    
    try:
        result = client.pareto(
            model="meta-llama/Llama-3.1-70B",
            gpu_cost=4.0,
            batch_size=8,
            seq_length=4096,
            training=True
        )
        
        print("\n✓ Pareto analysis complete")
        print_result(result, max_length=1000)
        
    except APIError as e:
        print(f"\n✗ Error: {e}")


def example_5_model_comparison(client: ParallelismAPIClient):
    """Example 5: Compare multiple models."""
    print_section("Example 5: Multi-Model Comparison")
    
    models = [
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-70B",
    ]
    
    print(f"Comparing models: {', '.join(models)}")
    
    try:
        result = client.compare(
            models=models,
            batch_size=1,
            seq_length=2048,
            training=True,
            mock_topology="h100",
            mock_gpus=8
        )
        
        print("\n✓ Comparison complete")
        print_result(result, max_length=1500)
        
    except APIError as e:
        print(f"\n✗ Error: {e}")


def example_6_inference_optimization(client: ParallelismAPIClient):
    """Example 6: Inference optimization."""
    print_section("Example 6: Inference Optimization")
    
    print("Optimizing Llama-3.1-70B for low-latency inference...")
    print("Target: <100ms latency per request")
    
    try:
        result = client.inference(
            model="meta-llama/Llama-3.1-70B",
            batch_size=1,
            seq_length=2048,
            max_output_length=512,
            latency_target_ms=100,
            mock_topology="h100",
            mock_gpus=2
        )
        
        print("\n✓ Inference optimization complete")
        print_result(result, max_length=1000)
        
    except APIError as e:
        print(f"\n✗ Error: {e}")


def example_7_vllm_setup(client: ParallelismAPIClient):
    """Example 7: vLLM configuration."""
    print_section("Example 7: vLLM Serving Configuration")
    
    print("Generating vLLM config for high-throughput serving...")
    print("Target: 10K tokens/sec throughput")
    
    try:
        result = client.vllm(
            model="meta-llama/Llama-3.1-70B",
            batch_size=64,
            max_tokens=512,
            throughput_target=10000,
            gpus=2
        )
        
        print("\n✓ vLLM configuration generated")
        print_result(result, max_length=1000)
        
    except APIError as e:
        print(f"\n✗ Error: {e}")


def example_8_troubleshooting(client: ParallelismAPIClient):
    """Example 8: Troubleshooting OOM errors."""
    print_section("Example 8: Troubleshooting CUDA OOM")
    
    print("Diagnosing 'CUDA out of memory' error...")
    
    try:
        result = client.troubleshoot(
            error_message="CUDA out of memory. Tried to allocate 2.00 GiB"
        )
        
        print("\n✓ Diagnosis complete")
        print_result(result, max_length=1500)
        
    except APIError as e:
        print(f"\n✗ Error: {e}")


def example_9_memory_breakdown(client: ParallelismAPIClient):
    """Example 9: Detailed memory breakdown."""
    print_section("Example 9: Memory Breakdown Analysis")
    
    print("Analyzing memory usage for Llama-3.1-70B with TP=2, PP=2...")
    
    try:
        result = client.memory(
            model="meta-llama/Llama-3.1-70B",
            batch_size=8,
            seq_length=2048,
            tp=2,
            pp=2,
            training=True
        )
        
        print("\n✓ Memory analysis complete")
        print_result(result, max_length=1000)
        
    except APIError as e:
        print(f"\n✗ Error: {e}")


def example_10_configuration_validation(client: ParallelismAPIClient):
    """Example 10: Validate configuration before training."""
    print_section("Example 10: Configuration Validation")
    
    print("Validating configuration: TP=2, PP=2, DP=4 on H100 80GB...")
    
    try:
        result = client.validate(
            model="meta-llama/Llama-3.1-70B",
            tp=2,
            pp=2,
            dp=4,
            memory=80,
            batch_size=1,
            seq_length=2048
        )
        
        print("\n✓ Validation complete")
        print_result(result, max_length=1000)
        
    except APIError as e:
        print(f"\n✗ Error: {e}")


def example_11_sharding_strategies(client: ParallelismAPIClient):
    """Example 11: Compare sharding strategies."""
    print_section("Example 11: Sharding Strategy Recommendations")
    
    print("Getting sharding recommendations for DP=8...")
    
    try:
        result = client.sharding(
            model="meta-llama/Llama-3.1-70B",
            dp=8,
            memory=80,
            batch_size=1,
            seq_length=2048,
            nodes=1,
            gpus=8
        )
        
        print("\n✓ Sharding recommendations received")
        print_result(result, max_length=1500)
        
    except APIError as e:
        print(f"\n✗ Error: {e}")


def example_12_model_analysis(client: ParallelismAPIClient):
    """Example 12: Detailed model architecture analysis."""
    print_section("Example 12: Model Architecture Analysis")
    
    print("Analyzing Llama-3.1-70B architecture...")
    
    try:
        result = client.analyze(
            model="meta-llama/Llama-3.1-70B",
            batch_size=8,
            seq_length=2048
        )
        
        print("\n✓ Model analysis complete")
        print_result(result, max_length=1000)
        
    except APIError as e:
        print(f"\n✗ Error: {e}")


def main():
    """Run all examples."""
    print("=" * 80)
    print(" Parallelism Strategy Advisor API - Usage Examples")
    print("=" * 80)
    
    # Create API client
    API_URL = "http://localhost:8000"
    print(f"\nConnecting to API at {API_URL}...")
    
    client = ParallelismAPIClient(API_URL, timeout=300)
    
    # Test connection
    try:
        health = client.health()
        print("✓ Connected to API successfully")
    except APIError as e:
        print(f"✗ Failed to connect to API: {e}")
        print("\nPlease ensure the API server is running:")
        print("  python parallelism_planner_server.py")
        sys.exit(1)
    
    # Run examples
    examples = [
        ("Basic Recommendations", example_1_basic_recommendations),
        ("Multi-Node Setup", example_2_multi_node_setup),
        ("Cost Analysis", example_3_cost_analysis),
        ("Pareto Analysis", example_4_pareto_analysis),
        ("Model Comparison", example_5_model_comparison),
        ("Inference Optimization", example_6_inference_optimization),
        ("vLLM Setup", example_7_vllm_setup),
        ("Troubleshooting", example_8_troubleshooting),
        ("Memory Breakdown", example_9_memory_breakdown),
        ("Configuration Validation", example_10_configuration_validation),
        ("Sharding Strategies", example_11_sharding_strategies),
        ("Model Analysis", example_12_model_analysis),
    ]
    
    for name, example_func in examples:
        try:
            example_func(client)
        except Exception as e:
            print(f"\n✗ Example '{name}' failed: {e}")
    
    print("\n" + "=" * 80)
    print(" Examples completed!")
    print("=" * 80)
    print("\nFor more information, visit http://localhost:8000/docs")


if __name__ == "__main__":
    main()

