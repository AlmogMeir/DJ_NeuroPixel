"""
Force drop and recreate the SessionTrial table with the correct schema.
Run this script to fix the missing start_time column issue.
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from BPOD.mazeBPOD import SessionTrial, schema

print("Current SessionTrial table structure:")
print(SessionTrial.describe())
print("\n" + "="*60 + "\n")

# Drop the SessionTrial table
print("Dropping SessionTrial table...")
SessionTrial.drop()
print("✓ SessionTrial table dropped")

print("\n" + "="*60 + "\n")

# Reimport to force recreation
print("Recreating SessionTrial table with updated schema...")
from importlib import reload
import BPOD.mazeBPOD as mazeBPOD_module
reload(mazeBPOD_module)
from BPOD.mazeBPOD import SessionTrial as SessionTrialNew

print("\nNew SessionTrial table structure:")
print(SessionTrialNew.describe())
print("\n✓ SessionTrial table recreated successfully!")
