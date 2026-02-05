#!/usr/bin/env python3
"""
Python Client Library for Parallelism Strategy Advisor API

This module provides a convenient Python interface for interacting with the
Parallelism Strategy Advisor API.

Usage:
    from parallelism_api_client import ParallelismAPIClient
    
    client = ParallelismAPIClient("http://localhost:8000")
    
    # Get recommendations
    result = client.recommend(
        model="meta-llama/Llama-3.1-70B",
        training=True,
        mock_topology="h100"
    )
    
    # Generate launch commands
    commands = client.launch(
        nodes=2, gpus=8, tp=2, pp=2, dp=4,
        sharding="zero3", script="train.py"
    )
"""

import requests
from typing import Optional, List, Dict, Any, Literal
from dataclasses import dataclass, asdict
import json


class APIError(Exception):
    """Exception raised for API errors."""
    def __init__(self, status_code: int, message: str, details: Optional[Dict] = None):
        self.status_code = status_code
        self.message = message
        self.details = details or {}
        super().__init__(f"API Error {status_code}: {message}")


@dataclass
class RecommendRequest:
    """Request for parallelism recommendations."""
    model: str
    batch_size: int = 1
    seq_length: int = 2048
    goal: Literal["throughput", "latency", "memory", "efficiency"] = "throughput"
    training: bool = False
    mock_topology: Optional[Literal["b200", "h100"]] = None
    mock_gpus: int = 4


@dataclass
class LaunchRequest:
    """Request for launch command generation."""
    nodes: int = 1
    gpus: int = 8
    tp: int = 1
    pp: int = 1
    dp: int = 8
    sharding: Literal["none", "zero1", "zero2", "zero3", "fsdp", "hsdp"] = "none"
    micro_batch: int = 1
    grad_accum: int = 1
    master_addr: str = "localhost"
    master_port: int = 29500
    script: str = "train.py"
    framework: Optional[Literal["torchrun", "deepspeed", "accelerate", "megatron"]] = None


class ParallelismAPIClient:
    """
    Client for interacting with Parallelism Strategy Advisor API.
    
    Attributes:
        base_url: Base URL of the API server
        timeout: Request timeout in seconds
        session: Requests session for connection pooling
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 300,
        verify_ssl: bool = True
    ):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API server
            timeout: Request timeout in seconds (default: 300)
            verify_ssl: Whether to verify SSL certificates (default: True)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.verify = verify_ssl
        
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make an API request.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request body data
            params: Query parameters
            
        Returns:
            Response data as dictionary
            
        Raises:
            APIError: If the API returns an error
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_data = response.json() if response.content else {}
                raise APIError(
                    status_code=response.status_code,
                    message=error_data.get('error', 'Unknown error'),
                    details=error_data
                )
                
        except requests.exceptions.RequestException as e:
            raise APIError(
                status_code=0,
                message=f"Request failed: {str(e)}",
                details={"exception": type(e).__name__}
            )
    
    def health(self) -> Dict[str, str]:
        """Check API health."""
        return self._request("GET", "/health")
    
    def presets(self) -> Dict[str, Any]:
        """List available model presets."""
        return self._request("GET", "/presets")
    
    def topology(
        self,
        mock: Optional[Literal["b200", "h100"]] = None,
        mock_gpus: int = 4
    ) -> Dict[str, Any]:
        """
        Get hardware topology.
        
        Args:
            mock: Use mock topology (b200 or h100)
            mock_gpus: Number of GPUs for mock topology
            
        Returns:
            Topology information
        """
        params = {}
        if mock:
            params["mock"] = mock
            params["mock_gpus"] = mock_gpus
        return self._request("GET", "/topology", params=params)
    
    def recommend(
        self,
        model: str,
        batch_size: int = 1,
        seq_length: int = 2048,
        goal: Literal["throughput", "latency", "memory", "efficiency"] = "throughput",
        training: bool = False,
        mock_topology: Optional[Literal["b200", "h100"]] = None,
        mock_gpus: int = 4
    ) -> Dict[str, Any]:
        """
        Get parallelism recommendations.
        
        Args:
            model: Model name or HuggingFace ID
            batch_size: Batch size
            seq_length: Sequence length
            goal: Optimization goal
            training: Configure for training
            mock_topology: Use mock topology
            mock_gpus: GPU count for mock topology
            
        Returns:
            Parallelism recommendations
        """
        data = {
            "model": model,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "goal": goal,
            "training": training,
            "mock_topology": mock_topology,
            "mock_gpus": mock_gpus
        }
        return self._request("POST", "/recommend", data=data)
    
    def sharding(
        self,
        model: str,
        dp: int = 8,
        memory: float = 80,
        batch_size: int = 1,
        seq_length: int = 2048,
        nodes: int = 1,
        gpus: int = 8
    ) -> Dict[str, Any]:
        """
        Get sharding strategy recommendations.
        
        Args:
            model: Model name or HuggingFace ID
            dp: Data parallel size
            memory: GPU memory in GB
            batch_size: Micro-batch size
            seq_length: Sequence length
            nodes: Number of nodes
            gpus: GPUs per node
            
        Returns:
            Sharding recommendations
        """
        data = {
            "model": model,
            "dp": dp,
            "memory": memory,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "nodes": nodes,
            "gpus": gpus
        }
        return self._request("POST", "/sharding", data=data)
    
    def launch(
        self,
        nodes: int = 1,
        gpus: int = 8,
        tp: int = 1,
        pp: int = 1,
        dp: int = 8,
        sharding: Literal["none", "zero1", "zero2", "zero3", "fsdp", "hsdp"] = "none",
        micro_batch: int = 1,
        grad_accum: int = 1,
        master_addr: str = "localhost",
        master_port: int = 29500,
        script: str = "train.py",
        framework: Optional[Literal["torchrun", "deepspeed", "accelerate", "megatron"]] = None
    ) -> Dict[str, Any]:
        """
        Generate framework launch commands.
        
        Args:
            nodes: Number of nodes
            gpus: GPUs per node
            tp: Tensor parallel size
            pp: Pipeline parallel size
            dp: Data parallel size
            sharding: Sharding strategy
            micro_batch: Micro-batch size
            grad_accum: Gradient accumulation steps
            master_addr: Master address
            master_port: Master port
            script: Training script
            framework: Specific framework (or None for all)
            
        Returns:
            Launch commands for frameworks
        """
        data = {
            "nodes": nodes,
            "gpus": gpus,
            "tp": tp,
            "pp": pp,
            "dp": dp,
            "sharding": sharding,
            "micro_batch": micro_batch,
            "grad_accum": grad_accum,
            "master_addr": master_addr,
            "master_port": master_port,
            "script": script,
            "framework": framework
        }
        return self._request("POST", "/launch", data=data)
    
    def pareto(
        self,
        model: str,
        gpu_cost: float = 4.0,
        batch_size: int = 8,
        seq_length: int = 4096,
        training: bool = False
    ) -> Dict[str, Any]:
        """
        Perform cost/throughput Pareto analysis.
        
        Args:
            model: Model name or HuggingFace ID
            gpu_cost: GPU hourly cost in USD
            batch_size: Batch size
            seq_length: Sequence length
            training: Analyze for training
            
        Returns:
            Pareto analysis results
        """
        data = {
            "model": model,
            "gpu_cost": gpu_cost,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "training": training
        }
        return self._request("POST", "/pareto", data=data)
    
    def analyze(
        self,
        model: str,
        batch_size: int = 8,
        seq_length: int = 2048
    ) -> Dict[str, Any]:
        """
        Analyze model architecture.
        
        Args:
            model: Model name or HuggingFace ID
            batch_size: Batch size for memory estimates
            seq_length: Sequence length
            
        Returns:
            Model architecture analysis
        """
        data = {
            "model": model,
            "batch_size": batch_size,
            "seq_length": seq_length
        }
        return self._request("POST", "/analyze", data=data)
    
    def estimate(
        self,
        model: str,
        tokens: int = 1_000_000_000_000,
        throughput: float = 100000,
        gpus: int = 8,
        gpu_cost: float = 4.0,
        checkpoint_interval: int = 1_000_000_000
    ) -> Dict[str, Any]:
        """
        Estimate training time and cost.
        
        Args:
            model: Model name or HuggingFace ID
            tokens: Total tokens to train
            throughput: Tokens per second
            gpus: Number of GPUs
            gpu_cost: GPU hourly cost in USD
            checkpoint_interval: Checkpoint every N tokens
            
        Returns:
            Training time and cost estimates
        """
        data = {
            "model": model,
            "tokens": tokens,
            "throughput": throughput,
            "gpus": gpus,
            "gpu_cost": gpu_cost,
            "checkpoint_interval": checkpoint_interval
        }
        return self._request("POST", "/estimate", data=data)
    
    def compare(
        self,
        models: List[str],
        batch_size: int = 1,
        seq_length: int = 2048,
        training: bool = False,
        mock_topology: Optional[Literal["b200", "h100"]] = None,
        mock_gpus: int = 4
    ) -> Dict[str, Any]:
        """
        Compare parallelism for multiple models.
        
        Args:
            models: List of model names to compare
            batch_size: Batch size
            seq_length: Sequence length
            training: Configure for training
            mock_topology: Use mock topology
            mock_gpus: GPU count for mock topology
            
        Returns:
            Comparison results
        """
        data = {
            "models": models,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "training": training,
            "mock_topology": mock_topology,
            "mock_gpus": mock_gpus
        }
        return self._request("POST", "/compare", data=data)
    
    def validate(
        self,
        model: str,
        tp: int = 1,
        pp: int = 1,
        dp: int = 8,
        memory: float = 80,
        batch_size: int = 1,
        seq_length: int = 2048
    ) -> Dict[str, Any]:
        """
        Validate parallelism configuration.
        
        Args:
            model: Model name or HuggingFace ID
            tp: Tensor parallel size
            pp: Pipeline parallel size
            dp: Data parallel size
            memory: GPU memory in GB
            batch_size: Batch size
            seq_length: Sequence length
            
        Returns:
            Validation results
        """
        data = {
            "model": model,
            "tp": tp,
            "pp": pp,
            "dp": dp,
            "memory": memory,
            "batch_size": batch_size,
            "seq_length": seq_length
        }
        return self._request("POST", "/validate", data=data)
    
    def inference(
        self,
        model: str,
        batch_size: int = 1,
        seq_length: int = 2048,
        max_output_length: int = 512,
        latency_target_ms: Optional[float] = None,
        throughput_target: Optional[float] = None,
        mock_topology: Optional[Literal["b200", "h100"]] = None,
        mock_gpus: int = 1
    ) -> Dict[str, Any]:
        """
        Get inference optimization recommendations.
        
        Args:
            model: Model name or HuggingFace ID
            batch_size: Batch size
            seq_length: Input sequence length
            max_output_length: Maximum output length
            latency_target_ms: Target latency in milliseconds
            throughput_target: Target throughput in tokens/sec
            mock_topology: Use mock topology
            mock_gpus: GPU count for mock topology
            
        Returns:
            Inference optimization recommendations
        """
        data = {
            "model": model,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "max_output_length": max_output_length,
            "latency_target_ms": latency_target_ms,
            "throughput_target": throughput_target,
            "mock_topology": mock_topology,
            "mock_gpus": mock_gpus
        }
        return self._request("POST", "/inference", data=data)
    
    def troubleshoot(
        self,
        error_message: Optional[str] = None,
        topic: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get troubleshooting help.
        
        Args:
            error_message: Error message to diagnose
            topic: Specific troubleshooting topic
            
        Returns:
            Troubleshooting information
        """
        data = {}
        if error_message:
            data["error_message"] = error_message
        if topic:
            data["topic"] = topic
        return self._request("POST", "/troubleshoot", data=data)
    
    def memory(
        self,
        model: str,
        batch_size: int = 8,
        seq_length: int = 2048,
        tp: int = 1,
        pp: int = 1,
        training: bool = True
    ) -> Dict[str, Any]:
        """
        Get memory breakdown analysis.
        
        Args:
            model: Model name or HuggingFace ID
            batch_size: Batch size
            seq_length: Sequence length
            tp: Tensor parallel size
            pp: Pipeline parallel size
            training: Training mode
            
        Returns:
            Memory breakdown
        """
        data = {
            "model": model,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "tp": tp,
            "pp": pp,
            "training": training
        }
        return self._request("POST", "/memory", data=data)
    
    def vllm(
        self,
        model: str,
        batch_size: int = 64,
        max_tokens: int = 512,
        latency_target_ms: Optional[float] = None,
        throughput_target: Optional[float] = None,
        gpus: int = 1
    ) -> Dict[str, Any]:
        """
        Get vLLM configuration.
        
        Args:
            model: Model name or HuggingFace ID
            batch_size: Batch size
            max_tokens: Maximum output tokens
            latency_target_ms: Target latency in milliseconds
            throughput_target: Target throughput in tokens/sec
            gpus: Number of GPUs
            
        Returns:
            vLLM configuration
        """
        data = {
            "model": model,
            "batch_size": batch_size,
            "max_tokens": max_tokens,
            "latency_target_ms": latency_target_ms,
            "throughput_target": throughput_target,
            "gpus": gpus
        }
        return self._request("POST", "/vllm", data=data)


# Convenience function
def create_client(base_url: str = "http://localhost:8000", **kwargs) -> ParallelismAPIClient:
    """
    Create a ParallelismAPIClient instance.
    
    Args:
        base_url: Base URL of the API server
        **kwargs: Additional arguments for ParallelismAPIClient
        
    Returns:
        ParallelismAPIClient instance
    """
    return ParallelismAPIClient(base_url, **kwargs)


# Example usage
if __name__ == "__main__":
    # Create client
    client = create_client()
    
    # Test health
    print("Testing API connection...")
    health = client.health()
    print(f"Health: {health}")
    
    # Get recommendations
    print("\nGetting parallelism recommendations...")
    result = client.recommend(
        model="meta-llama/Llama-3.1-70B",
        training=True,
        mock_topology="h100",
        mock_gpus=8
    )
    print(f"Recommendations: {json.dumps(result, indent=2)[:500]}...")
    
    # Generate launch commands
    print("\nGenerating launch commands...")
    commands = client.launch(
        nodes=2,
        gpus=8,
        tp=2,
        pp=2,
        dp=4,
        sharding="zero3",
        script="train.py"
    )
    print(f"Commands: {list(commands.get('commands', {}).keys())}")
    
    print("\n✓ All tests passed!")

