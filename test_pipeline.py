"""Test script to verify pipeline and API"""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.legal_kg.pipeline import LegalReasoningPipeline

print("Testing pipeline...")
p = LegalReasoningPipeline(Path("data"))
result = p.run("A danh B", top_k=1)
output = result.to_json_dict(include_candidates=True)
print("Result keys:", output.keys())
print("Crime:", output.get("result", {}).get("toi_danh_de_xuat"))
print("Candidates:", len(output.get("candidates", [])))
print("DONE!")
