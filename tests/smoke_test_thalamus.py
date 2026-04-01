# smoke_test_thalamus.py
from verification import ThalamusVerifier

verifier = ThalamusVerifier()

packet = {
    "memory_id": "mem_123",
    "claim": "Mount Everest is 8848 meters tall.",
    "verification_status": "unverified",
    "staleness_risk": 0.65,
    "memory_class": "numeric_fact",
    "source_count": 0,
    "hypothesis": False,
    "conflict_with_ids": [],
    "needs_external_verification": True,
}

intake = """
A reference sheet says Mount Everest is 8849 meters tall.
Another note says K2 is the second-highest mountain.
"""

result = verifier.verify(packet, intake_text=intake)
print("VERIFY RESULT:")
print(result)

fb = verifier.verify_and_build_feedback(packet, intake_text=intake)
print("\nFEEDBACK RESULT:")
print(fb)