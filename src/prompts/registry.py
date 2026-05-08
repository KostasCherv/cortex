"""Disk-backed prompt template registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import md5
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template


@dataclass
class PromptRegistry:
    """Load and cache prompt templates from disk."""

    template_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    _environment: Environment = field(init=False, repr=False)
    _cache: dict[str, Template] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._environment = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=False,
            keep_trailing_newline=True,
        )

    def render(self, name: str, context: dict[str, object]) -> tuple[str, str]:
        template_name = f"{name}.j2"
        template = self._cache.get(name)
        if template is None:
            template = self._environment.get_template(template_name)
            self._cache[name] = template

        template_path = self.template_dir / template_name
        template_source = template_path.read_text(encoding="utf-8")
        prompt_version = md5(template_source.encode("utf-8")).hexdigest()
        rendered = template.render(**context)
        return rendered, prompt_version


prompt_registry = PromptRegistry()
