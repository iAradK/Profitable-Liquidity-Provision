"""Load config.toml (settings) and secrets.toml (API key) into typed, frozen dataclasses."""

import tomllib
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.toml"
SECRETS_PATH = ROOT / "secrets.toml"

STEPS = ("introspect", "meta", "hourly", "daily", "positions", "prices", "report", "all")


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Pool:
    category: str
    label: str
    address: str


@dataclass(frozen=True)
class Paths:
    output_dir: Path
    daily_close_dir: Path

    @property
    def hourly(self) -> Path:
        return self.output_dir / "hourly"

    @property
    def daily(self) -> Path:
        return self.output_dir / "daily"

    @property
    def meta(self) -> Path:
        return self.output_dir / "meta"

    @property
    def positions(self) -> Path:
        return self.output_dir / "positions"

    @property
    def prices(self) -> Path:
        return self.output_dir / "token_prices"

    @property
    def reports(self) -> Path:
        return self.output_dir / "reports"

    @property
    def schema_file(self) -> Path:
        return self.meta / "schema.json"


@dataclass(frozen=True)
class RunConfig:
    step: str
    only_pools: tuple[str, ...]
    refresh_schema: bool


@dataclass(frozen=True)
class SubgraphConfig:
    host: str
    subgraph_id: str
    api_key: str

    @property
    def url(self) -> str:
        return f"{self.host.rstrip('/')}/api/{self.api_key}/subgraphs/id/{self.subgraph_id}"

    @property
    def safe_url(self) -> str:
        """URL without the API key, safe to log and save."""
        return f"{self.host.rstrip('/')}/api/<key>/subgraphs/id/{self.subgraph_id}"


@dataclass(frozen=True)
class NetworkConfig:
    max_workers: int
    max_retries: int
    min_interval_seconds: float
    timeout_seconds: float


@dataclass(frozen=True)
class DownloadConfig:
    start_ts: int
    end_ts: int  # exclusive
    page_size: int
    max_positions: int
    snapshot_chunk: int
    checkpoint_every: int
    force_refresh: bool


@dataclass(frozen=True)
class ReportConfig:
    max_workers: int


@dataclass(frozen=True)
class Config:
    run: RunConfig
    subgraph: SubgraphConfig
    network: NetworkConfig
    download: DownloadConfig
    report: ReportConfig
    paths: Paths
    pools: tuple[Pool, ...]

    def selected_pools(self) -> list[Pool]:
        if not self.run.only_pools:
            return list(self.pools)
        return [p for p in self.pools if p.address in self.run.only_pools]


def _timestamp(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def _read_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def _load_api_key(path: Path) -> str:
    if not path.exists():
        return ""
    return _read_toml(path).get("subgraph", {}).get("api_key", "")


def load_config(path: Path = CONFIG_PATH, secrets_path: Path = SECRETS_PATH) -> Config:
    try:
        raw = _read_toml(path)
        run, sub, dl, net, rep = (raw[k] for k in ("run", "subgraph", "download", "network", "report"))
        pools = tuple(Pool(p["category"], p["label"], p["address"].lower()) for p in raw["universe"]["pools"])
        config = Config(
            run=RunConfig(
                step=run["step"],
                only_pools=tuple(a.lower() for a in run["only_pools"]),
                refresh_schema=run["refresh_schema"],
            ),
            subgraph=SubgraphConfig(sub["host"], sub["subgraph_id"], _load_api_key(secrets_path)),
            network=NetworkConfig(
                max_workers=net["max_workers"],
                max_retries=net["max_retries"],
                min_interval_seconds=net["min_interval_seconds"],
                timeout_seconds=net["timeout_seconds"],
            ),
            download=DownloadConfig(
                start_ts=_timestamp(dl["start_date"]),
                end_ts=_timestamp(dl["end_date"]),
                page_size=dl["page_size"],
                max_positions=dl["max_positions"],
                snapshot_chunk=dl["snapshot_chunk"],
                checkpoint_every=dl["checkpoint_every"],
                force_refresh=dl["force_refresh"],
            ),
            report=ReportConfig(max_workers=rep["max_workers"]),
            paths=Paths(ROOT / dl["output_dir"], ROOT / rep["daily_close_dir"]),
            pools=pools,
        )
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {exc.filename}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path.name} is not valid TOML: {exc}") from exc
    except KeyError as exc:
        raise ConfigError(f"{path.name} is missing the setting {exc}") from exc
    _validate(config)
    return config


def _validate(config: Config) -> None:
    if config.run.step not in STEPS:
        raise ConfigError(f"run.step must be one of {', '.join(STEPS)}, got {config.run.step!r}")
    unknown = set(config.run.only_pools) - {p.address for p in config.pools}
    if unknown:
        raise ConfigError(f"run.only_pools has addresses not in [universe]: {', '.join(sorted(unknown))}")
    if config.download.start_ts >= config.download.end_ts:
        raise ConfigError("download.start_date must be before download.end_date")
    if config.network.max_workers < 1 or config.report.max_workers < 1:
        raise ConfigError("max_workers must be at least 1")
    if config.run.step != "report" and not config.subgraph.api_key:
        raise ConfigError(f"put your The Graph API key in {SECRETS_PATH.name} (copy secrets.example.toml)")
