"""
Hypothesis Framework
===================

Rigorous hypothesis tracking for UCCE research.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime


class EvidenceLevel(Enum):
    """Level of evidence supporting a hypothesis."""
    DEMONSTRATED = "demonstrated"      # Math/logic proof
    VALIDATED = "validated"            # Empirical validation
    PLAUSIBLE = "plausible"           # Theoretical support
    SPECULATIVE = "speculative"       # Interesting but unproven
    REFUTED = "refuted"              # Evidence against
    DEAD = "dead"                    # Failed critical tests


class HypothesisStatus(Enum):
    """Status of hypothesis in research pipeline."""
    PROPOSED = "proposed"
    TESTING = "testing"
    REFUTING = "refuting"
    VALIDATING = "validating"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ABANDONED = "abandoned"


@dataclass
class Hypothesis:
    """
    A research hypothesis with full documentation.
    
    Every hypothesis must have:
    - Clear claim
    - Theoretical justification
    - Expected gain
    - Known risks
    - Experimental design
    - Failure criteria
    """
    
    # Identity
    id: str
    title: str
    
    # The claim
    claim: str
    
    # Justification
    theoretical_justification: str
    literature_support: List[str] = field(default_factory=list)
    
    # Expected outcomes
    expected_compression_ratio: float = 1.0  # e.g., 10x, 100x
    expected_quality_preservation: float = 1.0  # 0.0 to 1.0
    expected_speedup: float = 1.0
    
    # Known risks
    known_risks: List[str] = field(default_factory=list)
    
    # Experimental design
    test_models: List[str] = field(default_factory=list)
    test_metrics: List[str] = field(default_factory=list)
    baseline_methods: List[str] = field(default_factory=list)
    
    # Failure criteria (what would prove this wrong)
    failure_criteria: List[str] = field(default_factory=list)
    
    # Current status
    evidence_level: EvidenceLevel = EvidenceLevel.SPECULATIVE
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    
    # Evidence
    supporting_evidence: List[Dict] = field(default_factory=list)
    refuting_evidence: List[Dict] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tested_at: Optional[datetime] = None
    
    # Results
    measured_compression: Optional[float] = None
    measured_quality: Optional[float] = None
    measured_speedup: Optional[float] = None
    experimental_notes: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "claim": self.claim,
            "evidence_level": self.evidence_level.value,
            "status": self.status.value,
            "expected_compression": self.expected_compression_ratio,
            "measured_compression": self.measured_compression,
            "measured_quality": self.measured_quality,
        }
    
    def update_status(self, status: HypothesisStatus):
        """Update hypothesis status."""
        self.status = status
        self.updated_at = datetime.now()
    
    def add_supporting_evidence(self, evidence: Dict):
        """Add supporting evidence."""
        evidence["timestamp"] = datetime.now().isoformat()
        self.supporting_evidence.append(evidence)
        self._update_evidence_level()
        self.updated_at = datetime.now()
    
    def add_refuting_evidence(self, evidence: Dict):
        """Add refuting evidence."""
        evidence["timestamp"] = datetime.now().isoformat()
        self.refuting_evidence.append(evidence)
        self._update_evidence_level()
        self.updated_at = datetime.now()
    
    def _update_evidence_level(self):
        """Update evidence level based on evidence."""
        if len(self.refuting_evidence) > 0:
            # Strong refutation
            if any(e.get("strength", 0) > 0.7 for e in self.refuting_evidence):
                self.evidence_level = EvidenceLevel.REFUTED
        elif len(self.supporting_evidence) > 0:
            # Has supporting evidence
            if all(e.get("strength", 0) > 0.7 for e in self.supporting_evidence):
                self.evidence_level = EvidenceLevel.VALIDATED


@dataclass
class Experiment:
    """A single experiment to test a hypothesis."""
    hypothesis_id: str
    name: str
    model_name: str
    metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    success: bool
    notes: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def improvement(self, metric: str) -> float:
        """Calculate improvement for a metric."""
        if metric not in self.metrics or metric not in self.baseline_metrics:
            return 0.0
        baseline = self.baseline_metrics[metric]
        if baseline == 0:
            return 0.0
        return (self.metrics[metric] - baseline) / baseline


class HypothesisLab:
    """
    Scientific laboratory for hypothesis management.
    
    Manages the full lifecycle:
    1. Proposal
    2. Design
    3. Testing
    4. Refutation attempt
    5. Validation
    6. Jury decision
    """
    
    def __init__(self):
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.experiments: List[Experiment] = []
        self.next_id: int = 1
    
    def create_hypothesis(
        self,
        title: str,
        claim: str,
        theoretical_justification: str,
        expected_compression: float = 1.0,
        known_risks: List[str] = None,
    ) -> Hypothesis:
        """
        Create a new hypothesis with full documentation.
        
        REQUIRED FIELDS:
        - title: Short descriptive title
        - claim: What this hypothesis claims
        - theoretical_justification: Why this should work
        - expected_compression: What compression ratio we expect
        - known_risks: What could go wrong
        """
        h_id = f"H{self.next_id:03d}"
        self.next_id += 1
        
        hypothesis = Hypothesis(
            id=h_id,
            title=title,
            claim=claim,
            theoretical_justification=theoretical_justification,
            expected_compression_ratio=expected_compression,
            known_risks=known_risks or [],
        )
        
        self.hypotheses[h_id] = hypothesis
        return hypothesis
    
    def get_hypothesis(self, h_id: str) -> Optional[Hypothesis]:
        """Get a hypothesis by ID."""
        return self.hypotheses.get(h_id)
    
    def get_all_hypotheses(self) -> List[Hypothesis]:
        """Get all hypotheses."""
        return list(self.hypotheses.values())
    
    def get_hypotheses_by_status(self, status: HypothesisStatus) -> List[Hypothesis]:
        """Get hypotheses by status."""
        return [h for h in self.hypotheses.values() if h.status == status]
    
    def get_hypotheses_by_level(self, level: EvidenceLevel) -> List[Hypothesis]:
        """Get hypotheses by evidence level."""
        return [h for h in self.hypotheses.values() if h.evidence_level == level]
    
    def run_experiment(
        self,
        hypothesis_id: str,
        name: str,
        model_name: str,
        metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        success: bool,
        notes: str = "",
    ) -> Experiment:
        """Record an experiment."""
        exp = Experiment(
            hypothesis_id=hypothesis_id,
            name=name,
            model_name=model_name,
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            success=success,
            notes=notes,
        )
        
        self.experiments.append(exp)
        
        # Update hypothesis
        h = self.get_hypothesis(hypothesis_id)
        if h:
            h.tested_at = datetime.now()
            h.measured_compression = metrics.get("compression_ratio")
            h.measured_quality = metrics.get("quality_preservation")
            h.measured_speedup = metrics.get("speedup")
            h.experimental_notes = notes
            
            if success:
                h.add_supporting_evidence({
                    "type": "experiment",
                    "name": name,
                    "strength": 0.8,
                    "metrics": metrics,
                })
            else:
                h.add_refuting_evidence({
                    "type": "experiment",
                    "name": name,
                    "strength": 0.8,
                    "notes": notes,
                })
        
        return exp
    
    def summary(self) -> str:
        """Generate a summary of all hypotheses."""
        lines = [
            "=" * 60,
            "HYPOTHESIS LAB SUMMARY",
            "=" * 60,
            "",
            f"Total Hypotheses: {len(self.hypotheses)}",
            f"Total Experiments: {len(self.experiments)}",
            "",
            "BY STATUS:",
        ]
        
        for status in HypothesisStatus:
            count = len(self.get_hypotheses_by_status(status))
            if count > 0:
                lines.append(f"  {status.value}: {count}")
        
        lines.extend(["", "BY EVIDENCE LEVEL:"])
        
        for level in EvidenceLevel:
            count = len(self.get_hypotheses_by_level(level))
            if count > 0:
                lines.append(f"  {level.value}: {count}")
        
        lines.extend(["", "ACTIVE HYPOTHESES:"])
        
        for h in self.hypotheses.values():
            if h.status in [HypothesisStatus.PROPOSED, HypothesisStatus.TESTING, HypothesisStatus.VALIDATING]:
                lines.append(f"  [{h.id}] {h.title}")
                lines.append(f"      Claim: {h.claim[:60]}...")
                lines.append(f"      Level: {h.evidence_level.value}")
        
        return "\n".join(lines)


# Example hypotheses for UCCE research

def create_ucc_hypotheses() -> HypothesisLab:
    """Create initial hypotheses for UCCE research."""
    lab = HypothesisLab()
    
    # H1: Quantization (BASELINE)
    lab.create_hypothesis(
        title="Quantization INT8",
        claim="INT8 quantization achieves 2x compression with <5% quality loss",
        theoretical_justification="Reduced precision arithmetic preserves most information. Established in literature (GPTQ, LLM.int8).",
        expected_compression=2.0,
        known_risks=["Quality loss at high compression", "Hardware support required"],
    ).evidence_level = EvidenceLevel.VALIDATED
    
    # H2: Quantization INT4 (BASELINE)
    lab.create_hypothesis(
        title="Quantization INT4",
        claim="INT4 quantization achieves 4x compression with <10% quality loss",
        theoretical_justification="4-bit precision sufficient for most weights. GPTQ/AWQ validate this.",
        expected_compression=4.0,
        known_risks=["Higher quality loss", "Dequantization overhead"],
    ).evidence_level = EvidenceLevel.VALIDATED
    
    # H3: Low-Rank Decomposition
    lab.create_hypothesis(
        title="Low-Rank Weight Decomposition",
        claim="SVD-based decomposition achieves 2-4x compression with <5% quality loss",
        theoretical_justification="Weight matrices often have low-rank structure. SVD captures principal components.",
        expected_compression=3.0,
        known_risks=["Not all layers have low-rank structure", "Quality loss varies by layer"],
    )
    
    # H4: Pruning
    lab.create_hypothesis(
        title="Magnitude-based Pruning",
        claim="Pruning 50% of weights achieves 2x compression with <10% quality loss",
        theoretical_justification="Small weights contribute less to output. Magnitude pruning removes noise.",
        expected_compression=2.0,
        known_risks=["Optimal sparsity varies by model", "Quality degradation with aggressive pruning"],
    ).evidence_level = EvidenceLevel.VALIDATED
    
    # H5: Vector Quantization
    lab.create_hypothesis(
        title="Vector Quantization for Embeddings",
        claim="VQ on embeddings achieves 8-16x compression with <15% quality loss",
        theoretical_justification="Embeddings have clustered structure. Codebook compression effective.",
        expected_compression=10.0,
        known_risks=["Codebook optimization complex", "Quality loss on rare tokens"],
    )
    
    # H6: Sparse MoE (SPECULATIVE)
    lab.create_hypothesis(
        title="Sparse Mixture of Experts",
        claim="MoE with 2/8 active experts achieves 4x effective compression",
        theoretical_justification="Only activate subset of experts per token. Mixtral validates approach.",
        expected_compression=4.0,
        known_risks=["Router overhead", "Expert load balancing"],
    ).evidence_level = EvidenceLevel.VALIDATED
    
    # H7: Hypernetwork Weight Generation (REFUTED in audit)
    lab.create_hypothesis(
        title="Hypernetwork Weight Generation",
        claim="A generator network can produce weights equivalent to trained weights",
        theoretical_justification="Hypernetworks learn to generate weights from seeds. Theoretical possibility.",
        expected_compression=1000.0,
        known_risks=[
            "Information-theoretic limit: generator must be as large as target",
            "Mode collapse common in hypernetworks",
            "No demonstration in literature",
        ],
    )
    lab.hypotheses["H007"].evidence_level = EvidenceLevel.REFUTED
    lab.hypotheses["H007"].add_refuting_evidence({
        "type": "theoretical",
        "strength": 1.0,
        "description": "Capacity argument: |G| >= |W| for lossless generation",
    })
    
    # H8: Fractal Compression (REFUTED in audit)
    lab.create_hypothesis(
        title="Fractal Weight Compression",
        claim="Weight matrices have fractal self-similarity enabling 4x+ compression",
        theoretical_justification="Hierarchical structure in transformers may show self-similarity.",
        expected_compression=4.0,
        known_risks=[
            "Measured correlation ~0.45, need 0.70+",
            "No proven fractal structure in weights",
            "Code returns noise (AUDIT FINDING)",
        ],
    )
    lab.hypotheses["H008"].evidence_level = EvidenceLevel.REFUTED
    lab.hypotheses["H008"].add_refuting_evidence({
        "type": "experimental",
        "strength": 0.9,
        "description": "FractalWeightCompression.decompress() returns torch.randn() - random noise",
    })
    
    # H9: Cognitive Seed Generation (SPECULATIVE - NEW)
    lab.create_hypothesis(
        title="Cognitive Seed + Rules = Intelligence",
        claim="A small seed + generative rules can reconstruct model capabilities",
        theoretical_justification="""
        Similar to procedural generation:
        - Minecraft: seed + rules = infinite world
        - Grammar: rules = infinite sentences
        - Physics: rules = infinite scenarios
        
        Perhaps neural networks learn rules that can be compressed.
        """,
        expected_compression=100.0,
        known_risks=[
            "No evidence rules are compressible",
            "Rules may be as complex as weights",
            "Training to learn rules is hard",
        ],
    )
    
    # H10: Concept Decomposition (SPECULATIVE - NEW)
    lab.create_hypothesis(
        title="Concept-based Model Decomposition",
        claim="Models can be decomposed into atomic concepts with 10x compression",
        theoretical_justification="""
        Hypothesis: Neural networks encode discrete concepts.
        If we can identify and separate concepts:
        - 'cat' concept
        - 'verb' concept  
        - 'Paris' concept
        Then compress each independently.
        """,
        expected_compression=10.0,
        known_risks=[
            "Concepts may be entangled (superposition)",
            "No algorithm to identify atomic concepts",
            "Concepts may not be discrete",
        ],
    )
    
    return lab
