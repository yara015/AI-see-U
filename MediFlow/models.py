"""
Hospital Resource Management - Pydantic Models

Defines Action, Observation, and State types for the OpenEnv hospital environment.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class PatientAssignment(BaseModel):
    """A single patient-to-resource assignment decision."""
    patient_id: int = Field(..., description="ID of patient to treat")
    doctor_id: int = Field(..., description="ID of doctor to assign")
    bed_id: int = Field(..., description="ID of bed to use")
    use_surgery: bool = Field(False, description="Whether this patient needs surgery")


class HospitalAction(BaseModel):
    """Action taken by the agent each time step."""
    assignments: List[PatientAssignment] = Field(
        default_factory=list,
        description="List of patient-to-resource assignments for this step",
    )
    reasoning: str = Field("", description="Agent reasoning for decisions")

    model_config = {"extra": "allow"}


class PatientInfo(BaseModel):
    """Patient information visible to the agent."""
    id: int
    severity: int = Field(..., ge=1, le=5, description="1=mild, 5=critical")
    specialization: str
    treatment_duration: int
    requires_surgery: bool
    medicine_needed: str
    wait_time: int = 0


class DoctorInfo(BaseModel):
    """Doctor status visible to the agent."""
    id: int
    specialization: str
    skill_level: float = Field(..., ge=0.5, le=1.0)
    is_available: bool


class SurgeonInfo(BaseModel):
    """Surgeon status visible to the agent."""
    id: int
    is_available: bool


class BedInfo(BaseModel):
    """Bed status visible to the agent."""
    id: int
    is_occupied: bool


class EpisodeStats(BaseModel):
    """Running statistics for the current episode."""
    patients_treated: int = 0
    total_arrivals: int = 0
    avg_wait_time: float = 0.0
    critical_treated: int = 0
    critical_total: int = 0
    resource_utilization: float = 0.0


class HospitalObservation(BaseModel):
    """Observation returned after each step with full hospital state."""
    current_step: int = 0
    max_steps: int = 48
    queue: List[PatientInfo] = Field(default_factory=list)
    doctors: List[DoctorInfo] = Field(default_factory=list)
    surgeons: List[SurgeonInfo] = Field(default_factory=list)
    beds: List[BedInfo] = Field(default_factory=list)
    medicines: Dict[str, int] = Field(default_factory=dict)
    stats: EpisodeStats = Field(default_factory=EpisodeStats)
    situation_report: str = Field("", description="Natural language summary")
    done: bool = False
    reward: float = 0.0
    metadata: Optional[Dict] = None

    model_config = {"extra": "allow"}


class HospitalState(BaseModel):
    """Internal environment state."""
    episode_id: str = ""
    step_count: int = 0
    task_id: str = ""
    stats: EpisodeStats = Field(default_factory=EpisodeStats)
