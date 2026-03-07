"""Read-only view over a completed sweep, grouped by config fingerprint."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields as dc_fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from .experiment import ExperimentResult


@dataclass(slots=True)
class SweepCase:
    """One executed case (a single repeat of an experiment)."""

    case_id: str
    case_dir: Path
    config_path: Path
    label: str
    repeat_number: int
    repeat_count: int
    sweep_values: dict[str, Any]

    def run_analysis(self, sweep_root: Path, **postpro: Any) -> ExperimentResult:
        """Run offline post-processing on this case's debate snapshot.

        Loads ``config.json`` from ``case_dir``, sets the experiment to
        offline/replay mode (``num_turns=0``, ``load_snapshot=True``,
        ``reuse_load_dir_for_outputs=True``), merges any *postpro* keyword
        overrides (e.g. ``semantic_analysis_metrics``, ``persona_analysis_metrics``),
        and delegates to :func:`~agora.experiment.run_persona_experiment`.

        Parameters
        ----------
        sweep_root:
            Absolute root of the sweep directory (i.e. ``SweepManifest.sweep_root``).
        **postpro:
            Any :class:`~agora.experiment.ExperimentConfig` fields you want to
            override for the analysis pass (analysis metrics, scoring model, …).
        """
        from .experiment import ExperimentConfig, build_experiment_config, run_persona_experiment

        abs_case_dir = sweep_root / self.case_dir
        abs_config_path = sweep_root / self.config_path

        base_raw = json.loads(abs_config_path.read_text(encoding="utf-8"))
        allowed = {f.name for f in dc_fields(ExperimentConfig)}
        # Strip fields we override explicitly, plus any machine-specific paths from the
        # original run (output_dir, outputs_root, index_csv, run_name).
        _OFFLINE_OVERRIDE_FIELDS = {
            "num_turns", "load_snapshot", "load_dir", "save_snapshot",
            "reuse_load_dir_for_outputs", "indexed_output",
            "output_dir", "outputs_root", "index_csv", "run_name",
            "catalog_path", "prompts_path",
        }
        base_config = {
            k: v for k, v in base_raw.items()
            if k in allowed and k not in _OFFLINE_OVERRIDE_FIELDS
        }

        offline_payload: dict[str, Any] = {
            **base_config,
            "num_turns": 0,
            "load_snapshot": True,
            "load_dir": abs_case_dir,
            "save_snapshot": False,
            "reuse_load_dir_for_outputs": True,
            "indexed_output": False,
            **postpro,
        }

        return run_persona_experiment(build_experiment_config(offline_payload))


@dataclass(slots=True)
class ExperimentGroup:
    """All repeats that share the same config fingerprint (i.e. the same experiment)."""

    config_fingerprint: str
    sweep_values: dict[str, Any]
    cases: list[SweepCase]

    @property
    def repeat_count(self) -> int:
        return len(self.cases)

    @property
    def case_ids(self) -> list[str]:
        return [c.case_id for c in self.cases]

    def abs_case_dirs(self, sweep_root: Path) -> list[Path]:
        """Absolute paths to each case directory."""
        return [sweep_root / c.case_dir for c in self.cases]

    def abs_config_paths(self, sweep_root: Path) -> list[Path]:
        """Absolute paths to each case config.json."""
        return [sweep_root / c.config_path for c in self.cases]

    def run_analysis(self, sweep_root: Path, **postpro: Any) -> list[ExperimentResult]:
        """Run offline post-processing for every repeat in this experiment group.

        Parameters
        ----------
        sweep_root:
            Absolute root of the sweep directory.
        **postpro:
            Forwarded verbatim to :meth:`SweepCase.run_analysis` for each repeat.
        """
        return [case.run_analysis(sweep_root, **postpro) for case in self.cases]


@dataclass(slots=True)
class SweepManifest:
    """
    Structured, read-only view of a sweep's manifest.json.

    Groups all cases by their ``config_fingerprint`` so that each
    :class:`ExperimentGroup` represents one unique experiment with one or more
    repeats.

    Typical usage::

        manifest = SweepManifest.from_path("outputs/sweeps/manifest.json")
        for group in manifest:
            print(group.config_fingerprint, group.repeat_count)
    """

    schema_version: int
    generated_at: str
    sweep_root: Path
    notes: str | None
    total_cases: int
    groups: list[ExperimentGroup]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_path(cls, path: Path | str) -> SweepManifest:
        """Load from a ``manifest.json`` file or the directory that contains it."""
        manifest_path = Path(path)
        if manifest_path.is_dir():
            manifest_path = manifest_path / "manifest.json"
        sweep_root = manifest_path.resolve().parent

        data: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))

        groups_by_fp: dict[str, ExperimentGroup] = {}
        for raw in data.get("cases", []):
            fp = raw["config_fingerprint"]
            if fp not in groups_by_fp:
                groups_by_fp[fp] = ExperimentGroup(
                    config_fingerprint=fp,
                    sweep_values=dict(raw.get("sweep_values", {})),
                    cases=[],
                )
            groups_by_fp[fp].cases.append(
                SweepCase(
                    case_id=raw["case_id"],
                    case_dir=Path(raw["case_dir"]),
                    config_path=Path(raw["config_path"]),
                    label=raw["label"],
                    repeat_number=raw["repeat_number"],
                    repeat_count=raw["repeat_count"],
                    sweep_values=dict(raw.get("sweep_values", {})),
                )
            )

        return cls(
            schema_version=data["schema_version"],
            generated_at=data["generated_at"],
            sweep_root=sweep_root,
            notes=data.get("notes"),
            total_cases=data["total_cases"],
            groups=list(groups_by_fp.values()),
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def experiment_count(self) -> int:
        """Number of unique experiments (distinct config fingerprints)."""
        return len(self.groups)

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self) -> Iterator[ExperimentGroup]:
        return iter(self.groups)

    def __getitem__(self, key: int | str) -> ExperimentGroup:
        """Access a group by integer index or config fingerprint string."""
        if isinstance(key, int):
            return self.groups[key]
        for group in self.groups:
            if group.config_fingerprint == key:
                return group
        raise KeyError(key)
