"""
Hospital Resource Management — Core Environment Logic.

Implements the simulation: patient arrivals (Poisson), doctor/surgeon/bed
allocation, medicine inventory, treatment completion, and reward computation.
"""

import math
import uuid
from typing import List, Dict, Optional

import numpy as np

import sys, os

_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SERVER_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from models import (
    HospitalAction,
    HospitalObservation,
    HospitalState,
    PatientInfo,
    DoctorInfo,
    SurgeonInfo,
    BedInfo,
    EpisodeStats,
    PatientAssignment,
)
from server.entities import Patient, Doctor, Surgeon, Bed, ActiveTreatment
from server.tasks import TaskConfig, TASK_CONFIGS, SPECIALIZATIONS, MEDICINE_TYPES
from server.graders import compute_grader_score


class HospitalEnvironment:
    """
    Hospital Resource Management simulation.

    The agent assigns patients to doctors/beds each step while managing
    scarce surgeons and medicine inventory under stochastic patient arrivals.
    """

    def __init__(self):
        self._cfg: Optional[TaskConfig] = None
        self._rng: Optional[np.random.Generator] = None
        self._step: int = 0
        self._episode_id: str = ""
        self._task_id: str = ""

        self._queue: List[Patient] = []
        self._doctors: List[Doctor] = []
        self._surgeons: List[Surgeon] = []
        self._beds: List[Bed] = []
        self._medicines: Dict[str, int] = {}
        self._treatments: List[ActiveTreatment] = []

        self._treated = 0
        self._arrivals = 0
        self._total_wait = 0.0
        self._crit_treated = 0
        self._crit_total = 0
        self._doc_busy = 0
        self._bed_busy = 0
        self._pid = 0
        self._cum_reward = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        task_id: Optional[str] = None,
    ) -> HospitalObservation:
        tid = task_id or "easy"
        if tid not in TASK_CONFIGS:
            tid = "easy"
        self._task_id = tid
        self._cfg = TASK_CONFIGS[tid]

        actual_seed = seed if seed is not None else self._cfg.seed
        self._rng = np.random.default_rng(actual_seed)

        self._step = 0
        self._episode_id = str(uuid.uuid4())[:8]
        self._pid = 0
        self._cum_reward = 0.0

        self._doctors = self._make_doctors()
        self._surgeons = [Surgeon(id=i) for i in range(self._cfg.num_surgeons)]
        self._beds = [Bed(id=i) for i in range(self._cfg.num_beds)]
        self._medicines = {m: self._cfg.medicine_capacity for m in MEDICINE_TYPES}
        self._queue = []
        self._treatments = []

        self._treated = 0
        self._arrivals = 0
        self._total_wait = 0.0
        self._crit_treated = 0
        self._crit_total = 0
        self._doc_busy = 0
        self._bed_busy = 0

        # Initial arrivals
        for _ in range(int(self._rng.poisson(self._cfg.arrival_rate))):
            self._queue.append(self._new_patient())

        return self._obs(0.0, False)

    def step(self, action: HospitalAction) -> HospitalObservation:
        if self._cfg is None:
            return self._obs(0.0, True)

        self._step += 1
        r = 0.0

        # 1) Complete finished treatments
        done_tx = [t for t in self._treatments if t.end_step <= self._step]
        for t in done_tx:
            self._treatments.remove(t)
            self._treated += 1
            self._total_wait += t.patient_wait_time
            if t.is_critical:
                self._crit_treated += 1
            for b in self._beds:
                if b.id == t.bed_id:
                    b.occupied = False
                    b.patient_id = None
            r += 0.1
            r += max(0, 0.05 * (1.0 - t.patient_wait_time / 10.0))
            if t.is_critical:
                r += 0.08 * (t.patient_severity / 5.0)

        # 2) Process assignments
        for a in action.assignments:
            r += self._assign(a)

        # 3) New arrivals
        for _ in range(int(self._rng.poisson(self._cfg.arrival_rate))):
            self._queue.append(self._new_patient())

        # 4) Wait penalties
        for p in self._queue:
            p.wait_time += 1
            r -= 0.01 * p.severity * p.wait_time / 5.0
            if p.severity >= 4 and p.wait_time > 3:
                r -= 0.15

        # 5) Replenish meds
        for m in self._medicines:
            self._medicines[m] = min(
                self._medicines[m] + self._cfg.medicine_replenish,
                self._cfg.medicine_capacity,
            )

        # 6) Idle penalty
        if self._queue:
            idle_d = sum(1 for d in self._doctors if d.available_at <= self._step)
            idle_b = sum(1 for b in self._beds if not b.occupied)
            if idle_d > 0 and idle_b > 0:
                r -= 0.01 * min(idle_d, idle_b)

        # 7) Utilization tracking
        self._doc_busy += sum(
            1 for d in self._doctors if d.available_at > self._step
        )
        self._bed_busy += sum(1 for b in self._beds if b.occupied)

        # 8) Cap queue
        if len(self._queue) > 50:
            self._queue.sort(key=lambda p: (-p.severity, -p.wait_time))
            over = self._queue[50:]
            self._queue = self._queue[:50]
            r -= 0.02 * len(over)

        self._cum_reward += r
        ep_done = self._step >= self._cfg.max_steps
        return self._obs(r, ep_done)

    @property
    def state(self) -> HospitalState:
        return HospitalState(
            episode_id=self._episode_id,
            step_count=self._step,
            task_id=self._task_id,
            stats=self._stats(),
        )

    def get_grader_score(self) -> float:
        return compute_grader_score(self._stats().model_dump())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assign(self, a: PatientAssignment) -> float:
        r = 0.0
        patient = next((p for p in self._queue if p.id == a.patient_id), None)
        if patient is None:
            return -0.05

        doctor = next((d for d in self._doctors if d.id == a.doctor_id), None)
        if doctor is None or doctor.available_at > self._step:
            return -0.05

        bed = next((b for b in self._beds if b.id == a.bed_id), None)
        if bed is None or bed.occupied:
            return -0.05

        surgeon_id = None
        if a.use_surgery or patient.requires_surgery:
            surgeon = next(
                (s for s in self._surgeons if s.available_at <= self._step), None
            )
            if surgeon is None:
                return -0.03
            surgeon_id = surgeon.id

        if patient.medicine_needed in self._medicines:
            if self._medicines[patient.medicine_needed] <= 0:
                r -= 0.02
            else:
                self._medicines[patient.medicine_needed] -= 1

        dur = patient.treatment_duration
        if doctor.specialization == patient.specialization:
            dur = max(1, math.ceil(dur * 0.7 / doctor.skill_level))
        else:
            dur = max(1, math.ceil(dur * 1.3 / doctor.skill_level))
            r -= 0.01

        doctor.available_at = self._step + dur
        bed.occupied = True
        bed.patient_id = patient.id
        bed.release_at = self._step + dur

        if surgeon_id is not None:
            for s in self._surgeons:
                if s.id == surgeon_id:
                    s.available_at = self._step + dur

        self._treatments.append(
            ActiveTreatment(
                patient_id=patient.id,
                patient_severity=patient.severity,
                doctor_id=doctor.id,
                bed_id=bed.id,
                surgeon_id=surgeon_id,
                start_step=self._step,
                end_step=self._step + dur,
                patient_wait_time=patient.wait_time,
                is_critical=patient.severity >= 4,
            )
        )
        self._queue.remove(patient)
        return r

    def _new_patient(self) -> Patient:
        self._pid += 1
        self._arrivals += 1
        is_crit = bool(self._rng.random() < self._cfg.critical_fraction)
        if is_crit:
            sev = int(self._rng.choice([4, 5]))
            self._crit_total += 1
        else:
            sev = int(self._rng.choice([1, 2, 3]))
        spec = str(self._rng.choice(SPECIALIZATIONS))
        surg = bool(self._rng.random() < self._cfg.surgery_fraction)
        dur = max(1, int(self._rng.integers(1, 4)) + sev // 2)
        if surg:
            dur += 2
        med = str(self._rng.choice(MEDICINE_TYPES))
        return Patient(
            id=self._pid,
            severity=sev,
            specialization=spec,
            treatment_duration=dur,
            arrival_time=self._step,
            requires_surgery=surg,
            medicine_needed=med,
        )

    def _make_doctors(self) -> List[Doctor]:
        docs = []
        for i in range(self._cfg.num_doctors):
            spec = SPECIALIZATIONS[i % len(SPECIALIZATIONS)]
            skill = round(0.6 + float(self._rng.random()) * 0.4, 2)
            docs.append(Doctor(id=i, specialization=spec, skill_level=skill))
        return docs

    def _stats(self) -> EpisodeStats:
        steps = max(self._step, 1)
        doc_slots = len(self._doctors) * steps
        bed_slots = len(self._beds) * steps
        avg_w = (self._total_wait / self._treated) if self._treated else 0.0
        d_util = self._doc_busy / max(doc_slots, 1)
        b_util = self._bed_busy / max(bed_slots, 1)
        return EpisodeStats(
            patients_treated=self._treated,
            total_arrivals=self._arrivals,
            avg_wait_time=round(avg_w, 2),
            critical_treated=self._crit_treated,
            critical_total=self._crit_total,
            resource_utilization=round((d_util + b_util) / 2, 4),
        )

    def _obs(self, reward: float, done: bool) -> HospitalObservation:
        q = [
            PatientInfo(
                id=p.id,
                severity=p.severity,
                specialization=p.specialization,
                treatment_duration=p.treatment_duration,
                requires_surgery=p.requires_surgery,
                medicine_needed=p.medicine_needed,
                wait_time=p.wait_time,
            )
            for p in self._queue
        ]
        docs = [
            DoctorInfo(
                id=d.id,
                specialization=d.specialization,
                skill_level=d.skill_level,
                is_available=d.available_at <= self._step,
            )
            for d in self._doctors
        ]
        surgs = [
            SurgeonInfo(id=s.id, is_available=s.available_at <= self._step)
            for s in self._surgeons
        ]
        beds = [BedInfo(id=b.id, is_occupied=b.occupied) for b in self._beds]
        st = self._stats()

        report = self._situation_report(st)

        return HospitalObservation(
            current_step=self._step,
            max_steps=self._cfg.max_steps if self._cfg else 48,
            queue=q,
            doctors=docs,
            surgeons=surgs,
            beds=beds,
            medicines=dict(self._medicines),
            stats=st,
            situation_report=report,
            done=done,
            reward=round(reward, 4),
        )

    def _situation_report(self, st: EpisodeStats) -> str:
        if self._cfg is None:
            return ""
        crit_in_q = sum(1 for p in self._queue if p.severity >= 4)
        avail_docs = sum(1 for d in self._doctors if d.available_at <= self._step)
        avail_beds = sum(1 for b in self._beds if not b.occupied)
        avail_surg = sum(1 for s in self._surgeons if s.available_at <= self._step)
        lines = [
            f"=== HOSPITAL STATUS (Step {self._step}/{self._cfg.max_steps}) ===",
            f"Queue: {len(self._queue)} patients ({crit_in_q} critical)",
            f"Doctors available: {avail_docs}/{len(self._doctors)}",
            f"Surgeons available: {avail_surg}/{len(self._surgeons)}",
            f"Beds available: {avail_beds}/{len(self._beds)}",
            f"Treated so far: {st.patients_treated}",
            f"Avg wait: {st.avg_wait_time:.1f} steps",
        ]
        low_meds = [m for m, v in self._medicines.items() if v <= 3]
        if low_meds:
            lines.append(f"LOW MEDICINE ALERT: {', '.join(low_meds)}")
        urgent = [p for p in self._queue if p.severity >= 4 and p.wait_time > 2]
        if urgent:
            ids = [str(p.id) for p in urgent[:5]]
            lines.append(f"URGENT: Critical patients waiting too long: IDs {', '.join(ids)}")
        return "\n".join(lines)
