#!/usr/bin/env python3
"""
FastAPI Server for Parallelism Strategy Advisor

Provides REST API access to all parallelism planning features:
- Parallelism recommendations (TP/PP/DP/CP/EP)
- Sharding strategies (ZeRO/FSDP/HSDP)
- Launch command generation (torchrun/deepspeed/accelerate)
- Cost/throughput Pareto analysis
- Calibration from benchmark data
- Model analysis and topology detection
- Inference optimization
- RLHF and MoE optimization
- And much more...

Usage:
    uvicorn parallelism_planner_server:app --host 0.0.0.0 --port 8000 --reload
    
    # Or with the script directly:
    python parallelism_planner_server.py
"""

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
import sys
import traceback
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

app = FastAPI(
    title="Parallelism Strategy Advisor API",
    description="Comprehensive API for distributed training parallelism optimization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# =============================================================================
# Pydantic Models for Request/Response
# =============================================================================

class GoalType(str, Enum):
    throughput = "throughput"
    latency = "latency"
    memory = "memory"
    efficiency = "efficiency"


class ShardingType(str, Enum):
    none = "none"
    zero1 = "zero1"
    zero2 = "zero2"
    zero3 = "zero3"
    fsdp = "fsdp"
    hsdp = "hsdp"


class FrameworkType(str, Enum):
    torchrun = "torchrun"
    deepspeed = "deepspeed"
    accelerate = "accelerate"
    megatron = "megatron"


class TopologyChoice(str, Enum):
    b200 = "b200"
    h100 = "h100"


class RecommendRequest(BaseModel):
    model: str = Field(..., description="Model name or HuggingFace ID", example="meta-llama/Llama-3.1-70B")
    batch_size: int = Field(1, description="Batch size", ge=1)
    seq_length: int = Field(2048, description="Sequence length", ge=1)
    goal: GoalType = Field(GoalType.throughput, description="Optimization goal")
    training: bool = Field(False, description="Configure for training")
    mock_topology: Optional[TopologyChoice] = Field(None, description="Use mock topology")
    mock_gpus: int = Field(4, description="GPU count for mock topology", ge=1)


class ShardingRequest(BaseModel):
    model: str = Field(..., description="Model name or HuggingFace ID")
    dp: int = Field(8, description="Data parallel size", ge=1)
    memory: float = Field(80, description="GPU memory in GB", gt=0)
    batch_size: int = Field(1, description="Micro-batch size", ge=1)
    seq_length: int = Field(2048, description="Sequence length", ge=1)
    nodes: int = Field(1, description="Number of nodes", ge=1)
    gpus: int = Field(8, description="GPUs per node", ge=1)


class LaunchRequest(BaseModel):
    nodes: int = Field(1, description="Number of nodes", ge=1)
    gpus: int = Field(8, description="GPUs per node", ge=1)
    tp: int = Field(1, description="Tensor parallel size", ge=1)
    pp: int = Field(1, description="Pipeline parallel size", ge=1)
    dp: int = Field(8, description="Data parallel size", ge=1)
    sharding: ShardingType = Field(ShardingType.none, description="Sharding strategy")
    micro_batch: int = Field(1, description="Micro-batch size", ge=1)
    grad_accum: int = Field(1, description="Gradient accumulation steps", ge=1)
    master_addr: str = Field("localhost", description="Master address")
    master_port: int = Field(29500, description="Master port", ge=1024, le=65535)
    script: str = Field("train.py", description="Training script")
    framework: Optional[FrameworkType] = Field(None, description="Output specific framework only")


class ParetoRequest(BaseModel):
    model: str = Field(..., description="Model name or HuggingFace ID")
    gpu_cost: float = Field(4.0, description="GPU hourly cost ($)", gt=0)
    batch_size: int = Field(8, description="Batch size", ge=1)
    seq_length: int = Field(4096, description="Sequence length", ge=1)
    training: bool = Field(False, description="Analyze for training")


class AnalyzeRequest(BaseModel):
    model: str = Field(..., description="Model name or HuggingFace ID")
    batch_size: int = Field(8, description="Batch size for memory estimates", ge=1)
    seq_length: int = Field(2048, description="Sequence length", ge=1)


class EstimateRequest(BaseModel):
    model: str = Field(..., description="Model name or HuggingFace ID")
    tokens: int = Field(1_000_000_000_000, description="Total tokens to train", ge=1)
    throughput: float = Field(100000, description="Tokens per second", gt=0)
    gpus: int = Field(8, description="Number of GPUs", ge=1)
    gpu_cost: float = Field(4.0, description="GPU hourly cost ($)", gt=0)
    checkpoint_interval: int = Field(1_000_000_000, description="Checkpoint every N tokens", ge=1)


class CompareRequest(BaseModel):
    models: List[str] = Field(..., description="Model names to compare", min_items=1)
    batch_size: int = Field(1, description="Batch size", ge=1)
    seq_length: int = Field(2048, description="Sequence length", ge=1)
    training: bool = Field(False, description="Configure for training")
    mock_topology: Optional[TopologyChoice] = Field(None, description="Use mock topology")
    mock_gpus: int = Field(4, description="GPU count for mock topology", ge=1)


class ValidateRequest(BaseModel):
    model: str = Field(..., description="Model name or HuggingFace ID")
    tp: int = Field(1, description="Tensor parallel size", ge=1)
    pp: int = Field(1, description="Pipeline parallel size", ge=1)
    dp: int = Field(8, description="Data parallel size", ge=1)
    memory: float = Field(80, description="GPU memory in GB", gt=0)
    batch_size: int = Field(1, description="Batch size", ge=1)
    seq_length: int = Field(2048, description="Sequence length", ge=1)


class OptimizeRequest(BaseModel):
    model: str = Field(..., description="Model name or HuggingFace ID")
    batch_size: int = Field(1, description="Batch size", ge=1)
    seq_length: int = Field(2048, description="Sequence length", ge=1)
    training: bool = Field(True, description="Configure for training")
    precision: Optional[str] = Field(None, description="Precision mode")
    checkpointing: Optional[str] = Field(None, description="Checkpointing strategy")
    mock_topology: Optional[TopologyChoice] = Field(None, description="Use mock topology")
    mock_gpus: int = Field(4, description="GPU count for mock topology", ge=1)


class BottleneckRequest(BaseModel):
    model: str = Field(..., description="Model name or HuggingFace ID")
    tp: int = Field(1, description="Tensor parallel size", ge=1)
    pp: int = Field(1, description="Pipeline parallel size", ge=1)
    dp: int = Field(8, description="Data parallel size", ge=1)
    batch_size: int = Field(8, description="Batch size", ge=1)
    seq_length: int = Field(2048, description="Sequence length", ge=1)


class ScalingRequest(BaseModel):
    model: str = Field(..., description="Model name or HuggingFace ID")
    base_gpus: int = Field(8, description="Base number of GPUs", ge=1)
    target_gpus: int = Field(64, description="Target number of GPUs", ge=1)
    batch_size: int = Field(8, description="Batch size", ge=1)
    seq_length: int = Field(2048, description="Sequence length", ge=1)


class WhatIfRequest(BaseModel):
    model: str = Field(..., description="Model name or HuggingFace ID")
    scenario: str = Field(..., description="What-if scenario", example="double_batch_size")
    current_tp: int = Field(1, description="Current TP size", ge=1)
    current_pp: int = Field(1, description="Current PP size", ge=1)
    current_dp: int = Field(8, description="Current DP size", ge=1)
    current_batch_size: int = Field(8, description="Current batch size", ge=1)


class BatchSizeRequest(BaseModel):
    model: str = Field(..., description="Model name or HuggingFace ID")
    memory: float = Field(80, description="GPU memory in GB", gt=0)
    seq_length: int = Field(2048, description="Sequence length", ge=1)
    tp: int = Field(1, description="Tensor parallel size", ge=1)
    pp: int = Field(1, description="Pipeline parallel size", ge=1)


class AutoTuneRequest(BaseModel):
    model: str = Field(..., description="Model name or HuggingFace ID")
    gpus: int = Field(8, description="Number of GPUs", ge=1)
    memory: float = Field(80, description="GPU memory in GB", gt=0)
    seq_length: int = Field(2048, description="Sequence length", ge=1)
    training: bool = Field(True, description="Configure for training")


class InferenceRequest(BaseModel):
    model: str = Field(..., description="Model name or HuggingFace ID")
    batch_size: int = Field(1, description="Batch size", ge=1)
    seq_length: int = Field(2048, description="Input sequence length", ge=1)
    max_output_length: int = Field(512, description="Max output length", ge=1)
    latency_target_ms: Optional[float] = Field(None, description="Target latency in ms", gt=0)
    throughput_target: Optional[float] = Field(None, description="Target throughput (tokens/sec)", gt=0)
    mock_topology: Optional[TopologyChoice] = Field(None, description="Use mock topology")
    mock_gpus: int = Field(1, description="GPU count for mock topology", ge=1)


class TroubleshootRequest(BaseModel):
    error_message: Optional[str] = Field(None, description="Error message to diagnose")
    topic: Optional[str] = Field(None, description="Troubleshooting topic")


class MemoryRequest(BaseModel):
    model: str = Field(..., description="Model name or HuggingFace ID")
    batch_size: int = Field(8, description="Batch size", ge=1)
    seq_length: int = Field(2048, description="Sequence length", ge=1)
    tp: int = Field(1, description="Tensor parallel size", ge=1)
    pp: int = Field(1, description="Pipeline parallel size", ge=1)
    training: bool = Field(True, description="Training mode")


class RLHFRequest(BaseModel):
    model: str = Field(..., description="Model name or HuggingFace ID")
    algorithm: str = Field("ppo", description="RLHF algorithm", pattern="^(ppo|dpo|rloo|reinforce)$")
    batch_size: int = Field(8, description="Batch size", ge=1)
    seq_length: int = Field(2048, description="Sequence length", ge=1)
    gpus: int = Field(8, description="Number of GPUs", ge=1)
    memory: float = Field(80, description="GPU memory in GB", gt=0)


class MoERequest(BaseModel):
    model: str = Field(..., description="Model name or HuggingFace ID")
    num_experts: int = Field(8, description="Number of experts", ge=1)
    experts_per_token: int = Field(2, description="Experts per token", ge=1)
    batch_size: int = Field(8, description="Batch size", ge=1)
    seq_length: int = Field(2048, description="Sequence length", ge=1)
    gpus: int = Field(8, description="Number of GPUs", ge=1)


class LongContextRequest(BaseModel):
    model: str = Field(..., description="Model name or HuggingFace ID")
    seq_length: int = Field(32768, description="Sequence length", ge=1024)
    batch_size: int = Field(1, description="Batch size", ge=1)
    gpus: int = Field(8, description="Number of GPUs", ge=1)
    memory: float = Field(80, description="GPU memory in GB", gt=0)


class VLLMRequest(BaseModel):
    model: str = Field(..., description="Model name or HuggingFace ID")
    batch_size: int = Field(64, description="Batch size", ge=1)
    max_tokens: int = Field(512, description="Max output tokens", ge=1)
    latency_target_ms: Optional[float] = Field(None, description="Target latency in ms", gt=0)
    throughput_target: Optional[float] = Field(None, description="Target throughput (tokens/sec)", gt=0)
    gpus: int = Field(1, description="Number of GPUs", ge=1)


class CommOverlapRequest(BaseModel):
    model: str = Field(..., description="Model name or HuggingFace ID")
    tp: int = Field(1, description="Tensor parallel size", ge=1)
    pp: int = Field(1, description="Pipeline parallel size", ge=1)
    dp: int = Field(8, description="Data parallel size", ge=1)
    batch_size: int = Field(8, description="Batch size", ge=1)


class ExportRequest(BaseModel):
    model: str = Field(..., description="Model name or HuggingFace ID")
    tp: int = Field(1, description="Tensor parallel size", ge=1)
    pp: int = Field(1, description="Pipeline parallel size", ge=1)
    dp: int = Field(8, description="Data parallel size", ge=1)
    sharding: ShardingType = Field(ShardingType.none, description="Sharding strategy")
    batch_size: int = Field(1, description="Batch size", ge=1)
    seq_length: int = Field(2048, description="Sequence length", ge=1)
    framework: str = Field("pytorch", description="Framework", pattern="^(pytorch|deepspeed|megatron)$")


class LargeScaleRequest(BaseModel):
    model: str = Field(..., description="Model name or HuggingFace ID")
    num_nodes: int = Field(16, description="Number of nodes", ge=1)
    gpus_per_node: int = Field(8, description="GPUs per node", ge=1)
    batch_size: int = Field(8, description="Batch size", ge=1)
    seq_length: int = Field(2048, description="Sequence length", ge=1)
    cluster_type: str = Field("nvlink_h100", description="Cluster type")


class RLOptimizationRequest(BaseModel):
    model: str = Field(..., description="Model name or HuggingFace ID")
    algorithm: str = Field("ppo", description="RL algorithm")
    ref_model_strategy: str = Field("separate", description="Reference model strategy")
    batch_size: int = Field(8, description="Batch size", ge=1)
    seq_length: int = Field(2048, description="Sequence length", ge=1)
    gpus: int = Field(8, description="Number of GPUs", ge=1)


class NCCLTuningRequest(BaseModel):
    num_nodes: int = Field(1, description="Number of nodes", ge=1)
    gpus_per_node: int = Field(8, description="GPUs per node", ge=1)
    network_type: str = Field("nvlink", description="Network type")


# =============================================================================
# Helper Functions
# =============================================================================

def _resolve_mock_topology(choice: str, mock_gpus: int = 4):
    """Resolve mock topology choice to actual topology object."""
    from core.parallelism_planner.providers.advisor import (
        create_mock_topology_b200_multigpu,
        create_mock_topology_h100_multigpu
    )
    
    if choice == "b200":
        return create_mock_topology_b200_multigpu(mock_gpus)
    if choice == "h100":
        return create_mock_topology_h100_multigpu(mock_gpus)
    raise ValueError("Use mock_topology='b200' or 'h100' to use a mock topology")


def handle_error(e: Exception) -> JSONResponse:
    """Handle errors and return appropriate JSON response."""
    error_msg = str(e)
    traceback_str = traceback.format_exc()
    
    return JSONResponse(
        status_code=500,
        content={
            "error": error_msg,
            "type": type(e).__name__,
            "traceback": traceback_str
        }
    )


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Parallelism Strategy Advisor API",
        "version": "1.0.0",
        "description": "Comprehensive API for distributed training parallelism optimization",
        "docs": "/docs",
        "endpoints": {
            "recommend": "POST /recommend - Get parallelism recommendations",
            "sharding": "POST /sharding - Get sharding strategy recommendations",
            "launch": "POST /launch - Generate framework launch commands",
            "pareto": "POST /pareto - Cost/throughput Pareto analysis",
            "analyze": "POST /analyze - Analyze model architecture",
            "estimate": "POST /estimate - Estimate training time and cost",
            "compare": "POST /compare - Compare multiple models",
            "validate": "POST /validate - Validate parallelism configuration",
            "optimize": "POST /optimize - Get advanced optimization recommendations",
            "bottleneck": "POST /bottleneck - Analyze performance bottlenecks",
            "scaling": "POST /scaling - Analyze scaling efficiency",
            "whatif": "POST /whatif - What-if scenario analysis",
            "batchsize": "POST /batchsize - Find maximum batch size",
            "autotune": "POST /autotune - Auto-tune configuration",
            "inference": "POST /inference - Inference optimization",
            "troubleshoot": "POST /troubleshoot - Troubleshooting guide",
            "memory": "POST /memory - Memory breakdown analysis",
            "rlhf": "POST /rlhf - RLHF optimization",
            "moe": "POST /moe - MoE optimization",
            "long_context": "POST /long-context - Long context optimization",
            "vllm": "POST /vllm - vLLM configuration",
            "comm_overlap": "POST /comm-overlap - Communication overlap analysis",
            "export": "POST /export - Export training configuration",
            "largescale": "POST /largescale - Large-scale cluster optimization",
            "presets": "GET /presets - List available model presets",
            "topology": "GET /topology - Get hardware topology",
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/presets")
async def get_presets():
    """List available model presets."""
    try:
        from core.parallelism_planner.providers.model_analyzer import ModelAnalyzer
        
        analyzer = ModelAnalyzer()
        presets = analyzer.list_presets()
        
        return {
            "presets": presets,
            "count": len(presets)
        }
    except Exception as e:
        return handle_error(e)


@app.get("/topology")
async def get_topology(
    mock: Optional[TopologyChoice] = Query(None, description="Use mock topology"),
    mock_gpus: int = Query(4, description="GPU count for mock topology", ge=1)
):
    """Detect or display hardware topology."""
    try:
        from core.parallelism_planner.providers.topology_detector import TopologyDetector
        
        detector = TopologyDetector()
        
        if mock:
            topology = _resolve_mock_topology(mock, mock_gpus)
        else:
            try:
                topology = detector.detect()
            except RuntimeError:
                topology = _resolve_mock_topology("b200", mock_gpus)
        
        return {
            "topology": topology.to_dict() if hasattr(topology, 'to_dict') else str(topology)
        }
    except Exception as e:
        return handle_error(e)


@app.post("/recommend")
async def recommend(request: RecommendRequest):
    """Get parallelism recommendations (TP/PP/DP/CP/EP)."""
    try:
        from core.parallelism_planner.providers.advisor import ParallelismAdvisor
        
        advisor = ParallelismAdvisor(auto_detect_topology=False)
        
        # Set topology
        if request.mock_topology:
            advisor.set_topology(_resolve_mock_topology(request.mock_topology, request.mock_gpus))
        else:
            try:
                advisor.detect_topology()
            except RuntimeError:
                advisor.set_topology(_resolve_mock_topology("b200", request.mock_gpus))
        
        result = advisor.recommend(
            model=request.model,
            batch_size=request.batch_size,
            seq_length=request.seq_length,
            goal=request.goal,
            is_training=request.training,
        )
        
        # Convert to JSON-serializable dict
        return result.to_dict() if hasattr(result, 'to_dict') else {"result": str(result)}
    except Exception as e:
        return handle_error(e)


@app.post("/sharding")
async def sharding(request: ShardingRequest):
    """Get ZeRO/FSDP/HSDP sharding recommendations."""
    try:
        from core.parallelism_planner.providers.model_analyzer import ModelAnalyzer
        from core.parallelism_planner.providers.sharding_strategies import ShardingOptimizer
        
        analyzer = ModelAnalyzer()
        model = analyzer.analyze(request.model)
        
        optimizer = ShardingOptimizer()
        recommendations = optimizer.recommend(
            model=model,
            dp_size=request.dp,
            gpu_memory_gb=request.memory,
            batch_size=request.batch_size,
            seq_length=request.seq_length,
            num_nodes=request.nodes,
            gpus_per_node=request.gpus,
        )
        
        return recommendations.to_dict() if hasattr(recommendations, 'to_dict') else {"result": str(recommendations)}
    except Exception as e:
        return handle_error(e)


@app.post("/launch")
async def launch(request: LaunchRequest):
    """Generate framework launch commands."""
    try:
        from core.parallelism_planner.providers.launch_commands import (
            LaunchCommandGenerator, LaunchConfig, ShardingStrategy
        )
        
        # Map sharding string to enum
        sharding_map = {
            "none": ShardingStrategy.NO_SHARD,
            "zero1": ShardingStrategy.ZERO_1,
            "zero2": ShardingStrategy.ZERO_2,
            "zero3": ShardingStrategy.ZERO_3,
            "fsdp": ShardingStrategy.FSDP_FULL,
            "hsdp": ShardingStrategy.HSDP,
        }
        sharding = sharding_map.get(request.sharding.value.lower(), ShardingStrategy.NO_SHARD)
        
        config = LaunchConfig(
            num_nodes=request.nodes,
            gpus_per_node=request.gpus,
            tp_size=request.tp,
            pp_size=request.pp,
            dp_size=request.dp,
            sharding=sharding,
            micro_batch_size=request.micro_batch,
            gradient_accumulation_steps=request.grad_accum,
            master_addr=request.master_addr,
            master_port=request.master_port,
        )
        
        gen = LaunchCommandGenerator()
        all_commands = gen.generate_all(config, request.script)
        
        if request.framework:
            fw = request.framework.value.lower()
            if fw in all_commands:
                return {
                    "framework": fw,
                    "commands": all_commands[fw]
                }
            else:
                raise HTTPException(status_code=404, detail=f"Framework {fw} not found")
        
        return {"commands": all_commands}
    except Exception as e:
        return handle_error(e)


@app.post("/pareto")
async def pareto(request: ParetoRequest):
    """Cost/throughput Pareto analysis."""
    try:
        from core.parallelism_planner.providers.advisor import ParallelismAdvisor
        from core.parallelism_planner.providers.pareto_analysis import ParetoAnalyzer, ConfigurationPoint
        
        # Get parallelism recommendations
        advisor = ParallelismAdvisor(auto_detect_topology=False)
        try:
            advisor.detect_topology()
        except RuntimeError:
            advisor.set_topology(_resolve_mock_topology("b200"))
        
        result = advisor.recommend(
            model=request.model,
            batch_size=request.batch_size,
            seq_length=request.seq_length,
            is_training=request.training,
        )
        
        # Convert to ConfigurationPoints
        configs = []
        for rec in result.recommendations:
            s = rec.strategy
            a = rec.analysis
            configs.append(ConfigurationPoint(
                name=f"TP{s.tp}_PP{s.pp}_DP{s.dp}" + (f"_CP{s.cp}" if s.cp > 1 else ""),
                tp=s.tp,
                pp=s.pp,
                dp=s.dp,
                throughput_tps=a.estimated_throughput_tps,
                latency_ms=a.estimated_latency_ms,
                memory_per_gpu_gb=a.memory_per_gpu_gb,
                num_gpus=s.world_size,
            ))
        
        pareto = ParetoAnalyzer(gpu_hourly_cost=request.gpu_cost)
        analysis = pareto.generate_cost_throughput_analysis(configs)
        viz = pareto.generate_visualization_data(configs)
        
        return {
            "analysis": analysis,
            "visualization": viz
        }
    except Exception as e:
        return handle_error(e)


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """Analyze model architecture."""
    try:
        from core.parallelism_planner.providers.model_analyzer import ModelAnalyzer
        
        analyzer = ModelAnalyzer()
        model = analyzer.analyze(request.model)
        
        # Generate memory estimates
        memory_analysis = analyzer.estimate_memory(
            model,
            batch_size=request.batch_size,
            seq_length=request.seq_length,
        )
        
        return {
            "model": model.to_dict() if hasattr(model, 'to_dict') else str(model),
            "memory_analysis": memory_analysis
        }
    except Exception as e:
        return handle_error(e)


@app.post("/estimate")
async def estimate(request: EstimateRequest):
    """Estimate training time and cost."""
    try:
        from core.parallelism_planner.providers.model_analyzer import ModelAnalyzer
        
        analyzer = ModelAnalyzer()
        model = analyzer.analyze(request.model)
        
        # Calculate training time
        training_time_hours = request.tokens / request.throughput / 3600
        training_cost = training_time_hours * request.gpus * request.gpu_cost
        
        # Calculate checkpoints
        num_checkpoints = request.tokens // request.checkpoint_interval
        checkpoint_size_gb = model.total_params * 4 / (1024 ** 3)  # FP32 size
        total_checkpoint_storage_gb = num_checkpoints * checkpoint_size_gb
        
        return {
            "model": request.model,
            "total_tokens": request.tokens,
            "throughput_tps": request.throughput,
            "num_gpus": request.gpus,
            "training_time_hours": round(training_time_hours, 2),
            "training_time_days": round(training_time_hours / 24, 2),
            "total_cost_usd": round(training_cost, 2),
            "cost_per_gpu_usd": round(training_cost / request.gpus, 2),
            "num_checkpoints": num_checkpoints,
            "checkpoint_size_gb": round(checkpoint_size_gb, 2),
            "total_checkpoint_storage_gb": round(total_checkpoint_storage_gb, 2),
            "gpu_hourly_cost": request.gpu_cost,
        }
    except Exception as e:
        return handle_error(e)


@app.post("/compare")
async def compare(request: CompareRequest):
    """Compare parallelism for multiple models."""
    try:
        from core.parallelism_planner.providers.advisor import ParallelismAdvisor
        
        advisor = ParallelismAdvisor(auto_detect_topology=False)
        
        # Set topology
        if request.mock_topology:
            advisor.set_topology(_resolve_mock_topology(request.mock_topology, request.mock_gpus))
        else:
            try:
                advisor.detect_topology()
            except RuntimeError:
                advisor.set_topology(_resolve_mock_topology("b200", request.mock_gpus))
        
        results = {}
        for model in request.models:
            result = advisor.recommend(
                model=model,
                batch_size=request.batch_size,
                seq_length=request.seq_length,
                is_training=request.training,
            )
            results[model] = result.to_dict() if hasattr(result, 'to_dict') else str(result)
        
        return {"comparison": results}
    except Exception as e:
        return handle_error(e)


@app.post("/validate")
async def validate(request: ValidateRequest):
    """Validate parallelism configuration."""
    try:
        from core.parallelism_planner.providers.validation import ConfigValidator
        from core.parallelism_planner.providers.model_analyzer import ModelAnalyzer
        
        analyzer = ModelAnalyzer()
        model = analyzer.analyze(request.model)
        
        validator = ConfigValidator()
        result = validator.validate(
            model=model,
            tp=request.tp,
            pp=request.pp,
            dp=request.dp,
            gpu_memory_gb=request.memory,
            batch_size=request.batch_size,
            seq_length=request.seq_length,
        )
        
        return result.to_dict() if hasattr(result, 'to_dict') else {"result": str(result)}
    except Exception as e:
        return handle_error(e)


@app.post("/optimize")
async def optimize(request: OptimizeRequest):
    """Get advanced optimization recommendations."""
    try:
        from core.parallelism_planner.providers.advanced_optimizations import get_advanced_optimization_report
        from core.parallelism_planner.providers.advisor import ParallelismAdvisor
        
        advisor = ParallelismAdvisor(auto_detect_topology=False)
        
        # Set topology
        if request.mock_topology:
            advisor.set_topology(_resolve_mock_topology(request.mock_topology, request.mock_gpus))
        else:
            try:
                advisor.detect_topology()
            except RuntimeError:
                advisor.set_topology(_resolve_mock_topology("b200", request.mock_gpus))
        
        result = advisor.recommend(
            model=request.model,
            batch_size=request.batch_size,
            seq_length=request.seq_length,
            is_training=request.training,
        )
        
        # Get advanced optimizations
        optimization_report = get_advanced_optimization_report(
            model=request.model,
            batch_size=request.batch_size,
            seq_length=request.seq_length,
            is_training=request.training,
        )
        
        return {
            "parallelism": result.to_dict() if hasattr(result, 'to_dict') else str(result),
            "optimizations": optimization_report
        }
    except Exception as e:
        return handle_error(e)


@app.post("/bottleneck")
async def bottleneck(request: BottleneckRequest):
    """Analyze performance bottlenecks."""
    try:
        from core.parallelism_planner.providers.bottleneck_analysis import analyze_bottlenecks
        from core.parallelism_planner.providers.model_analyzer import ModelAnalyzer
        
        analyzer = ModelAnalyzer()
        model = analyzer.analyze(request.model)
        
        analysis = analyze_bottlenecks(
            model=model,
            tp=request.tp,
            pp=request.pp,
            dp=request.dp,
            batch_size=request.batch_size,
            seq_length=request.seq_length,
        )
        
        return analysis.to_dict() if hasattr(analysis, 'to_dict') else {"result": str(analysis)}
    except Exception as e:
        return handle_error(e)


@app.post("/scaling")
async def scaling(request: ScalingRequest):
    """Analyze scaling efficiency."""
    try:
        from core.parallelism_planner.providers.bottleneck_analysis import analyze_scaling
        from core.parallelism_planner.providers.model_analyzer import ModelAnalyzer
        
        analyzer = ModelAnalyzer()
        model = analyzer.analyze(request.model)
        
        analysis = analyze_scaling(
            model=model,
            base_gpus=request.base_gpus,
            target_gpus=request.target_gpus,
            batch_size=request.batch_size,
            seq_length=request.seq_length,
        )
        
        return analysis.to_dict() if hasattr(analysis, 'to_dict') else {"result": str(analysis)}
    except Exception as e:
        return handle_error(e)


@app.post("/whatif")
async def whatif(request: WhatIfRequest):
    """What-if scenario analysis."""
    try:
        from core.parallelism_planner.providers.bottleneck_analysis import analyze_whatif
        from core.parallelism_planner.providers.model_analyzer import ModelAnalyzer
        
        analyzer = ModelAnalyzer()
        model = analyzer.analyze(request.model)
        
        analysis = analyze_whatif(
            model=model,
            scenario=request.scenario,
            current_tp=request.current_tp,
            current_pp=request.current_pp,
            current_dp=request.current_dp,
            current_batch_size=request.current_batch_size,
        )
        
        return analysis.to_dict() if hasattr(analysis, 'to_dict') else {"result": str(analysis)}
    except Exception as e:
        return handle_error(e)


@app.post("/batchsize")
async def batchsize(request: BatchSizeRequest):
    """Find maximum batch size."""
    try:
        from core.parallelism_planner.providers.auto_tuning import find_max_batch_size
        from core.parallelism_planner.providers.model_analyzer import ModelAnalyzer
        
        analyzer = ModelAnalyzer()
        model = analyzer.analyze(request.model)
        
        result = find_max_batch_size(
            model=model,
            gpu_memory_gb=request.memory,
            seq_length=request.seq_length,
            tp=request.tp,
            pp=request.pp,
        )
        
        return result.to_dict() if hasattr(result, 'to_dict') else {"result": str(result)}
    except Exception as e:
        return handle_error(e)


@app.post("/autotune")
async def autotune(request: AutoTuneRequest):
    """Auto-tune configuration."""
    try:
        from core.parallelism_planner.providers.auto_tuning import auto_tune_config
        from core.parallelism_planner.providers.model_analyzer import ModelAnalyzer
        
        analyzer = ModelAnalyzer()
        model = analyzer.analyze(request.model)
        
        result = auto_tune_config(
            model=model,
            num_gpus=request.gpus,
            gpu_memory_gb=request.memory,
            seq_length=request.seq_length,
            is_training=request.training,
        )
        
        return result.to_dict() if hasattr(result, 'to_dict') else {"result": str(result)}
    except Exception as e:
        return handle_error(e)


@app.post("/inference")
async def inference(request: InferenceRequest):
    """Inference optimization recommendations."""
    try:
        from core.parallelism_planner.providers.inference_optimization import get_inference_optimization_report
        from core.parallelism_planner.providers.advisor import ParallelismAdvisor
        
        advisor = ParallelismAdvisor(auto_detect_topology=False)
        
        # Set topology
        if request.mock_topology:
            advisor.set_topology(_resolve_mock_topology(request.mock_topology, request.mock_gpus))
        else:
            try:
                advisor.detect_topology()
            except RuntimeError:
                advisor.set_topology(_resolve_mock_topology("b200", request.mock_gpus))
        
        report = get_inference_optimization_report(
            model=request.model,
            batch_size=request.batch_size,
            input_length=request.seq_length,
            output_length=request.max_output_length,
            latency_target_ms=request.latency_target_ms,
            throughput_target_tps=request.throughput_target,
        )
        
        return report
    except Exception as e:
        return handle_error(e)


@app.post("/troubleshoot")
async def troubleshoot(request: TroubleshootRequest):
    """Troubleshooting guide."""
    try:
        from core.parallelism_planner.providers.troubleshooting import (
            diagnose_error, get_all_troubleshooting_topics
        )
        
        if request.error_message:
            result = diagnose_error(request.error_message)
            return {"diagnosis": result}
        elif request.topic:
            topics = get_all_troubleshooting_topics()
            if request.topic in topics:
                return {"topic": request.topic, "info": topics[request.topic]}
            else:
                return {"error": f"Topic '{request.topic}' not found", "available_topics": list(topics.keys())}
        else:
            topics = get_all_troubleshooting_topics()
            return {"topics": list(topics.keys())}
    except Exception as e:
        return handle_error(e)


@app.post("/memory")
async def memory(request: MemoryRequest):
    """Memory breakdown analysis."""
    try:
        from core.parallelism_planner.providers.troubleshooting import get_memory_breakdown
        from core.parallelism_planner.providers.model_analyzer import ModelAnalyzer
        
        analyzer = ModelAnalyzer()
        model = analyzer.analyze(request.model)
        
        breakdown = get_memory_breakdown(
            model=model,
            batch_size=request.batch_size,
            seq_length=request.seq_length,
            tp=request.tp,
            pp=request.pp,
            is_training=request.training,
        )
        
        return breakdown.to_dict() if hasattr(breakdown, 'to_dict') else {"result": str(breakdown)}
    except Exception as e:
        return handle_error(e)


@app.post("/rlhf")
async def rlhf(request: RLHFRequest):
    """RLHF optimization recommendations."""
    try:
        from core.parallelism_planner.providers.distributed_training import RLHFMemoryCalculator
        from core.parallelism_planner.providers.model_analyzer import ModelAnalyzer
        
        analyzer = ModelAnalyzer()
        model = analyzer.analyze(request.model)
        
        calculator = RLHFMemoryCalculator()
        result = calculator.calculate(
            model=model,
            algorithm=request.algorithm,
            batch_size=request.batch_size,
            seq_length=request.seq_length,
            num_gpus=request.gpus,
            gpu_memory_gb=request.memory,
        )
        
        return result.to_dict() if hasattr(result, 'to_dict') else {"result": str(result)}
    except Exception as e:
        return handle_error(e)


@app.post("/moe")
async def moe(request: MoERequest):
    """MoE (Mixture of Experts) optimization."""
    try:
        from core.parallelism_planner.providers.distributed_training import MoEOptimizer
        from core.parallelism_planner.providers.model_analyzer import ModelAnalyzer
        
        analyzer = ModelAnalyzer()
        model = analyzer.analyze(request.model)
        
        optimizer = MoEOptimizer()
        result = optimizer.optimize(
            model=model,
            num_experts=request.num_experts,
            experts_per_token=request.experts_per_token,
            batch_size=request.batch_size,
            seq_length=request.seq_length,
            num_gpus=request.gpus,
        )
        
        return result.to_dict() if hasattr(result, 'to_dict') else {"result": str(result)}
    except Exception as e:
        return handle_error(e)


@app.post("/long-context")
async def long_context(request: LongContextRequest):
    """Long context optimization."""
    try:
        from core.parallelism_planner.providers.distributed_training import LongContextOptimizer
        from core.parallelism_planner.providers.model_analyzer import ModelAnalyzer
        
        analyzer = ModelAnalyzer()
        model = analyzer.analyze(request.model)
        
        optimizer = LongContextOptimizer()
        result = optimizer.optimize(
            model=model,
            seq_length=request.seq_length,
            batch_size=request.batch_size,
            num_gpus=request.gpus,
            gpu_memory_gb=request.memory,
        )
        
        return result.to_dict() if hasattr(result, 'to_dict') else {"result": str(result)}
    except Exception as e:
        return handle_error(e)


@app.post("/vllm")
async def vllm(request: VLLMRequest):
    """vLLM configuration and optimization."""
    try:
        from core.parallelism_planner.providers.vllm_optimization import get_vllm_optimization
        
        result = get_vllm_optimization(
            model=request.model,
            batch_size=request.batch_size,
            max_tokens=request.max_tokens,
            latency_target_ms=request.latency_target_ms,
            throughput_target_tps=request.throughput_target,
            num_gpus=request.gpus,
        )
        
        return result
    except Exception as e:
        return handle_error(e)


@app.post("/comm-overlap")
async def comm_overlap(request: CommOverlapRequest):
    """Communication overlap analysis."""
    try:
        from core.parallelism_planner.providers.distributed_training import CommunicationOverlapAnalyzer
        from core.parallelism_planner.providers.model_analyzer import ModelAnalyzer
        
        analyzer = ModelAnalyzer()
        model = analyzer.analyze(request.model)
        
        overlap_analyzer = CommunicationOverlapAnalyzer()
        result = overlap_analyzer.analyze(
            model=model,
            tp=request.tp,
            pp=request.pp,
            dp=request.dp,
            batch_size=request.batch_size,
        )
        
        return result.to_dict() if hasattr(result, 'to_dict') else {"result": str(result)}
    except Exception as e:
        return handle_error(e)


@app.post("/export")
async def export(request: ExportRequest):
    """Export training configuration."""
    try:
        from core.parallelism_planner.providers.config_export import export_training_config
        from core.parallelism_planner.providers.model_analyzer import ModelAnalyzer
        
        analyzer = ModelAnalyzer()
        model = analyzer.analyze(request.model)
        
        config = export_training_config(
            model=model,
            tp=request.tp,
            pp=request.pp,
            dp=request.dp,
            sharding=request.sharding.value,
            batch_size=request.batch_size,
            seq_length=request.seq_length,
            framework=request.framework,
        )
        
        return config.to_dict() if hasattr(config, 'to_dict') else {"config": config}
    except Exception as e:
        return handle_error(e)


@app.post("/largescale")
async def largescale(request: LargeScaleRequest):
    """Large-scale cluster optimization."""
    try:
        from core.parallelism_planner.providers.large_scale_optimization import get_large_scale_optimization
        
        result = get_large_scale_optimization(
            model=request.model,
            num_nodes=request.num_nodes,
            gpus_per_node=request.gpus_per_node,
            batch_size=request.batch_size,
            seq_length=request.seq_length,
            cluster_type=request.cluster_type,
        )
        
        return result
    except Exception as e:
        return handle_error(e)


@app.post("/rl")
async def rl_optimization(request: RLOptimizationRequest):
    """RL/RLHF optimization (deep integration)."""
    try:
        from core.parallelism_planner.providers.rl_optimization import get_rl_optimization
        
        result = get_rl_optimization(
            model=request.model,
            algorithm=request.algorithm,
            ref_model_strategy=request.ref_model_strategy,
            batch_size=request.batch_size,
            seq_length=request.seq_length,
            num_gpus=request.gpus,
        )
        
        return result
    except Exception as e:
        return handle_error(e)


@app.post("/nccl")
async def nccl_tuning(request: NCCLTuningRequest):
    """NCCL tuning recommendations."""
    try:
        from core.parallelism_planner.providers.troubleshooting import get_nccl_tuning
        
        result = get_nccl_tuning(
            num_nodes=request.num_nodes,
            gpus_per_node=request.gpus_per_node,
            network_type=request.network_type,
        )
        
        return {"nccl_config": result}
    except Exception as e:
        return handle_error(e)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 80)
    print("Parallelism Strategy Advisor API Server")
    print("=" * 80)
    print("\nStarting server on http://0.0.0.0:8000")
    print("API Documentation: http://0.0.0.0:8000/docs")
    print("Alternative docs: http://0.0.0.0:8000/redoc")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 80)
    
    uvicorn.run(
        "parallelism_planner_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

