"""
Task difficulty configurations for the hospital environment.
Three tasks: easy, medium, hard — each with different resource/demand parameters.
"""

from dataclasses import dataclass

SPECIALIZATIONS = ["general", "cardiology", "orthopedics", "neurology", "pediatrics"]
MEDICINE_TYPES = ["antibiotics", "painkillers", "anesthetics", "blood_supply", "surgical_kit"]


@dataclass
class TaskConfig:
    task_id: str
    description: str
    max_steps: int
    num_doctors: int
    num_surgeons: int
    num_beds: int
    arrival_rate: float
    critical_fraction: float
    surgery_fraction: float
    medicine_capacity: int
    medicine_replenish: int
    seed: int


TASK_CONFIGS = {
    "easy": TaskConfig(
        task_id="easy",
        description="Easy: Manageable patient load with ample resources",
        max_steps=24,
        num_doctors=6,
        num_surgeons=3,
        num_beds=15,
        arrival_rate=1.5,
        critical_fraction=0.10,
        surgery_fraction=0.05,
        medicine_capacity=50,
        medicine_replenish=5,
        seed=42,
    ),
    "medium": TaskConfig(
        task_id="medium",
        description="Medium: Higher volume, fewer resources, more critical cases",
        max_steps=48,
        num_doctors=5,
        num_surgeons=2,
        num_beds=10,
        arrival_rate=2.5,
        critical_fraction=0.25,
        surgery_fraction=0.15,
        medicine_capacity=30,
        medicine_replenish=2,
        seed=123,
    ),
    "hard": TaskConfig(
        task_id="hard",
        description="Hard: Extreme surge, scarce resources, frequent surgeries",
        max_steps=48,
        num_doctors=4,
        num_surgeons=1,
        num_beds=8,
        arrival_rate=4.0,
        critical_fraction=0.40,
        surgery_fraction=0.30,
        medicine_capacity=15,
        medicine_replenish=1,
        seed=7,
    ),
}
