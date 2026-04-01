"""
Executive layer: goal selection + priority weighting (prefrontal-ish control).

This module is intentionally lightweight and safe to plug into existing Clair v2 flow.
"""
from .goal_manager import GoalManager
from .priority_manager import PriorityManager