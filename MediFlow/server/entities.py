"""
Internal entity dataclasses for the hospital simulation.
These are NOT exposed to the client — they represent internal state.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Patient:
    id: int
    severity: int            # 1 (mild) to 5 (critical)
    specialization: str      # e.g. "cardiology"
    treatment_duration: int  # time steps needed
    arrival_time: int
    requires_surgery: bool
    medicine_needed: str
    wait_time: int = 0


@dataclass
class Doctor:
    id: int
    specialization: str
    skill_level: float       # 0.5–1.0
    available_at: int = 0    # step when free


@dataclass
class Surgeon:
    id: int
    available_at: int = 0


@dataclass
class Bed:
    id: int
    occupied: bool = False
    patient_id: Optional[int] = None
    release_at: int = 0


@dataclass
class ActiveTreatment:
    patient_id: int
    patient_severity: int
    doctor_id: int
    bed_id: int
    surgeon_id: Optional[int]
    start_step: int
    end_step: int
    patient_wait_time: int
    is_critical: bool
