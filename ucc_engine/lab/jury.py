"""
Jury Module
==========

Scientific jury for hypothesis evaluation.

Every innovation receives evaluation on:
- Originality
- Compression achieved
- Intelligence preservation
- Feasibility
- Complexity

Decision: Abandon, Improve, or Integrate
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class JuryDecision(Enum):
    """Decision made by the jury."""
    ABANDON = "abandon"          # Drop this hypothesis
    IMPROVE = "improve"         # Needs more work
    INTEGRATE = "integrate"      # Ready for integration
    MONITOR = "monitor"         # Interesting but needs observation


@dataclass
class JuryScore:
    """Scores given by a jury member."""
    originality: float  # 0-10, how novel
    compression: float  # 0-10, compression achieved
    preservation: float  # 0-10, quality preserved
    feasibility: float  # 0-10, how practical
    complexity: float  # 0-10, inverse (lower = simpler)


@dataclass
class JuryVerdict:
    """Final verdict from the jury."""
    hypothesis_id: str
    hypothesis_name: str
    
    # Scores
    average_scores: JuryScore
    min_scores: JuryScore
    max_scores: JuryScore
    
    # Decision
    decision: JuryDecision
    confidence: float  # 0-1, how confident
    
    # Rationale
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    
    # Voting
    votes_abandon: int
    votes_improve: int
    votes_integrate: int
    votes_monitor: int
    
    def summary(self) -> str:
        """Generate a summary."""
        lines = [
            f"Jury Verdict: {self.hypothesis_name}",
            "=" * 50,
            f"Decision: {self.decision.value.upper()}",
            f"Confidence: {self.confidence:.0%}",
            "",
            "Scores (0-10):",
            f"  Originality: {self.average_scores.originality:.1f}",
            f"  Compression: {self.average_scores.compression:.1f}",
            f"  Preservation: {self.average_scores.preservation:.1f}",
            f"  Feasibility: {self.average_scores.feasibility:.1f}",
            f"  Complexity: {self.average_scores.complexity:.1f}",
            "",
            "Voting:",
            f"  Abandon: {self.votes_abandon}",
            f"  Improve: {self.votes_improve}",
            f"  Integrate: {self.votes_integrate}",
            f"  Monitor: {self.votes_monitor}",
            "",
            "Strengths:",
        ]
        for s in self.strengths[:3]:
            lines.append(f"  + {s}")
        
        if self.weaknesses:
            lines.append("\nWeaknesses:")
            for w in self.weaknesses[:3]:
                lines.append(f"  - {w}")
        
        return "\n".join(lines)


class JuryMember:
    """A single jury member with a perspective."""
    
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.votes: List[JuryDecision] = []
    
    def evaluate(self, hypothesis) -> JuryScore:
        """
        Evaluate a hypothesis.
        
        Each member has their own evaluation criteria.
        """
        raise NotImplementedError


class CompressionExpert(JuryMember):
    """Jury member focused on compression metrics."""
    
    def __init__(self):
        super().__init__("Compression Expert", "Compression")
    
    def evaluate(self, hypothesis) -> JuryScore:
        """Focus on compression ratio and efficiency."""
        compression = hypothesis.measured_compression or hypothesis.expected_compression_ratio
        
        # Higher is better for compression
        compression_score = min(10, compression)
        
        # Originality moderate
        originality = 5.0
        
        # Preservation depends on measured quality
        preservation = hypothesis.measured_quality or 0.7
        
        # Feasibility based on known risks
        feasibility = 8.0 if len(hypothesis.known_risks) < 2 else 5.0
        
        # Complexity - compression methods tend to be moderate
        complexity = 5.0
        
        return JuryScore(
            originality=originality,
            compression=compression_score,
            preservation=preservation * 10,
            feasibility=feasibility,
            complexity=complexity,
        )


class TheoryExpert(JuryMember):
    """Jury member focused on theoretical foundations."""
    
    def __init__(self):
        super().__init__("Theory Expert", "Theory")
    
    def evaluate(self, hypothesis) -> JuryScore:
        """Focus on theoretical justification."""
        # Check evidence level
        evidence_scores = {
            "demonstrated": 10.0,
            "validated": 8.0,
            "plausible": 6.0,
            "speculative": 4.0,
            "refuted": 2.0,
            "dead": 1.0,
        }
        feasibility = evidence_scores.get(hypothesis.evidence_level.value, 5.0)
        
        # Originality high for speculative ideas
        originality = 7.0 if hypothesis.evidence_level.value in ["speculative", "plausible"] else 5.0
        
        # Compression moderate
        compression = 5.0
        
        # Preservation depends on evidence
        preservation = 5.0
        
        # Complexity depends on known risks
        complexity = 4.0 + len(hypothesis.known_risks) * 0.5
        
        return JuryScore(
            originality=originality,
            compression=compression,
            preservation=preservation,
            feasibility=feasibility,
            complexity=min(10, complexity),
        )


class SystemsExpert(JuryMember):
    """Jury member focused on practical systems."""
    
    def __init__(self):
        super().__init__("Systems Expert", "Systems")
    
    def evaluate(self, hypothesis) -> JuryScore:
        """Focus on practical implementation."""
        # Feasibility high for validated methods
        evidence_feasibility = {
            "demonstrated": 9.0,
            "validated": 7.0,
            "plausible": 5.0,
            "speculative": 3.0,
            "refuted": 1.0,
            "dead": 0.5,
        }
        feasibility = evidence_feasibility.get(hypothesis.evidence_level.value, 5.0)
        
        # Originality low for established methods
        originality = 3.0 if hypothesis.evidence_level.value in ["demonstrated", "validated"] else 6.0
        
        # Compression depends on expectation
        compression = min(10, hypothesis.expected_compression_ratio)
        
        # Preservation high for validated
        preservation = 8.0 if hypothesis.evidence_level.value in ["demonstrated", "validated"] else 5.0
        
        # Complexity moderate
        complexity = 5.0
        
        return JuryScore(
            originality=originality,
            compression=compression,
            preservation=preservation,
            feasibility=feasibility,
            complexity=complexity,
        )


class Jury:
    """
    Scientific jury for evaluating hypotheses.
    
    Multiple experts evaluate each hypothesis.
    """
    
    def __init__(self):
        self.members: List[JuryMember] = [
            CompressionExpert(),
            TheoryExpert(),
            SystemsExpert(),
        ]
        self.verdicts: Dict[str, JuryVerdict] = {}
    
    def deliberate(self, hypothesis) -> JuryVerdict:
        """
        Have the jury deliberate on a hypothesis.
        
        Returns a final verdict.
        """
        scores_list: List[JuryScore] = []
        
        # Each member evaluates
        for member in self.members:
            score = member.evaluate(hypothesis)
            scores_list.append(score)
        
        # Aggregate scores
        avg_scores = self._average_scores(scores_list)
        min_scores = self._min_scores(scores_list)
        max_scores = self._max_scores(scores_list)
        
        # Voting
        votes = self._vote(scores_list)
        
        # Determine decision
        decision = self._make_decision(
            avg_scores,
            hypothesis.evidence_level.value,
            hypothesis.measured_compression or 1.0,
        )
        
        # Strengths and weaknesses
        strengths, weaknesses = self._analyze_strengths_weaknesses(
            hypothesis, avg_scores
        )
        
        # Recommendations
        recommendations = self._make_recommendations(
            hypothesis, avg_scores, decision
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(scores_list)
        
        verdict = JuryVerdict(
            hypothesis_id=hypothesis.id if hasattr(hypothesis, 'id') else "UNKNOWN",
            hypothesis_name=hypothesis.title if hasattr(hypothesis, 'title') else "Unknown",
            average_scores=avg_scores,
            min_scores=min_scores,
            max_scores=max_scores,
            decision=decision,
            confidence=confidence,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            votes_abandon=votes["abandon"],
            votes_improve=votes["improve"],
            votes_integrate=votes["integrate"],
            votes_monitor=votes["monitor"],
        )
        
        self.verdicts[hypothesis.id if hasattr(hypothesis, 'id') else "UNKNOWN"] = verdict
        return verdict
    
    def _average_scores(self, scores_list: List[JuryScore]) -> JuryScore:
        """Calculate average scores."""
        n = len(scores_list)
        return JuryScore(
            originality=sum(s.originality for s in scores_list) / n,
            compression=sum(s.compression for s in scores_list) / n,
            preservation=sum(s.preservation for s in scores_list) / n,
            feasibility=sum(s.feasibility for s in scores_list) / n,
            complexity=sum(s.complexity for s in scores_list) / n,
        )
    
    def _min_scores(self, scores_list: List[JuryScore]) -> JuryScore:
        """Calculate minimum scores."""
        return JuryScore(
            originality=min(s.originality for s in scores_list),
            compression=min(s.compression for s in scores_list),
            preservation=min(s.preservation for s in scores_list),
            feasibility=min(s.feasibility for s in scores_list),
            complexity=min(s.complexity for s in scores_list),
        )
    
    def _max_scores(self, scores_list: List[JuryScore]) -> JuryScore:
        """Calculate maximum scores."""
        return JuryScore(
            originality=max(s.originality for s in scores_list),
            compression=max(s.compression for s in scores_list),
            preservation=max(s.preservation for s in scores_list),
            feasibility=max(s.feasibility for s in scores_list),
            complexity=max(s.complexity for s in scores_list),
        )
    
    def _vote(self, scores_list: List[JuryScore]) -> Dict[str, int]:
        """Count votes from each member."""
        votes = {"abandon": 0, "improve": 0, "integrate": 0, "monitor": 0}
        
        for score in scores_list:
            # Decision logic
            if score.feasibility < 3 or score.preservation < 3:
                votes["abandon"] += 1
            elif score.feasibility > 7 and score.compression > 5:
                votes["integrate"] += 1
            elif score.originality > 7 and score.feasibility > 5:
                votes["monitor"] += 1
            else:
                votes["improve"] += 1
        
        return votes
    
    def _make_decision(
        self,
        avg_scores: JuryScore,
        evidence_level: str,
        measured_compression: float,
    ) -> JuryDecision:
        """Make final decision based on scores."""
        # Refuted or dead = abandon
        if evidence_level in ["refuted", "dead"]:
            return JuryDecision.ABANDON
        
        # Abandon if feasibility or preservation very low
        if avg_scores.feasibility < 3 or avg_scores.preservation < 3:
            return JuryDecision.ABANDON
        
        # Integrate if high scores across the board
        if (avg_scores.feasibility > 7 and 
            avg_scores.compression > 6 and 
            avg_scores.preservation > 7):
            return JuryDecision.INTEGRATE
        
        # Monitor if very original but unproven
        if avg_scores.originality > 8 and avg_scores.feasibility < 5:
            return JuryDecision.MONITOR
        
        # Default to improve
        return JuryDecision.IMPROVE
    
    def _analyze_strengths_weaknesses(
        self,
        hypothesis,
        avg_scores: JuryScore,
    ) -> tuple:
        """Analyze strengths and weaknesses."""
        strengths = []
        weaknesses = []
        
        # Compression
        if avg_scores.compression > 7:
            strengths.append(f"Strong compression: {avg_scores.compression:.1f}/10")
        elif avg_scores.compression < 4:
            weaknesses.append(f"Weak compression: {avg_scores.compression:.1f}/10")
        
        # Preservation
        if avg_scores.preservation > 7:
            strengths.append(f"High quality preservation: {avg_scores.preservation:.1f}/10")
        elif avg_scores.preservation < 4:
            weaknesses.append(f"Poor quality preservation: {avg_scores.preservation:.1f}/10")
        
        # Feasibility
        if avg_scores.feasibility > 7:
            strengths.append("Well-established theoretical foundation")
        elif avg_scores.feasibility < 4:
            weaknesses.append("Weak theoretical justification")
        
        # Originality
        if avg_scores.originality > 7:
            strengths.append("Highly original approach")
        
        return strengths, weaknesses
    
    def _make_recommendations(
        self,
        hypothesis,
        avg_scores: JuryScore,
        decision: JuryDecision,
    ) -> List[str]:
        """Make recommendations based on evaluation."""
        recs = []
        
        if decision == JuryDecision.ABANDON:
            recs.append("Abandon this hypothesis - insufficient evidence")
            recs.append("Focus on validated approaches instead")
        
        elif decision == JuryDecision.INTEGRATE:
            recs.append("Ready for integration into system")
            recs.append("Add comprehensive benchmarks")
        
        elif decision == JuryDecision.IMPROVE:
            recs.append("Needs more validation and testing")
            if avg_scores.preservation < 6:
                recs.append("Focus on improving quality preservation")
            if avg_scores.compression < 6:
                recs.append("Explore higher compression ratios")
        
        elif decision == JuryDecision.MONITOR:
            recs.append("Interesting but risky - monitor for breakthroughs")
            recs.append("Could lead to paradigm shift if successful")
        
        return recs
    
    def _calculate_confidence(self, scores_list: List[JuryScore]) -> float:
        """Calculate confidence based on agreement."""
        if len(scores_list) < 2:
            return 0.5
        
        # Calculate variance in scores
        avg = self._average_scores(scores_list)
        
        variance = 0.0
        for score in scores_list:
            diff = (
                (score.originality - avg.originality) ** 2 +
                (score.compression - avg.compression) ** 2 +
                (score.preservation - avg.preservation) ** 2 +
                (score.feasibility - avg.feasibility) ** 2
            )
            variance += diff
        
        variance /= len(scores_list)
        
        # Lower variance = higher confidence
        confidence = max(0.1, 1.0 - variance / 100)
        return confidence
    
    def summary(self) -> str:
        """Generate a summary of all verdicts."""
        lines = [
            "=" * 60,
            "JURY VERDICTS SUMMARY",
            "=" * 60,
            "",
        ]
        
        for h_id, verdict in self.verdicts.items():
            lines.append("")
            lines.append(f"[{verdict.decision.value.upper()}] {verdict.hypothesis_name}")
            lines.append(f"  Confidence: {verdict.confidence:.0%}")
            lines.append(f"  Scores: Comp={verdict.average_scores.compression:.1f}, "
                       f"Pres={verdict.average_scores.preservation:.1f}, "
                       f"Feas={verdict.average_scores.feasibility:.1f}")
        
        return "\n".join(lines)
