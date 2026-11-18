"""Records every cleaning rule and check so the data card is generated, never hand-written."""
from dataclasses import dataclass, field


@dataclass
class Ledger:
    rules: list = field(default_factory=list)    # (id, table, description, rows_affected)
    checks: list = field(default_factory=list)   # (id, severity, passed, detail)
    facts: dict = field(default_factory=dict)    # named numbers quoted in the data card

    def rule(self, rid: str, table: str, desc: str, rows: int | str):
        self.rules.append((rid, table, desc, rows))

    def check(self, cid: str, passed: bool, detail: str, severity: str = "error"):
        self.checks.append((cid, severity, bool(passed), detail))

    def fact(self, key: str, value):
        self.facts[key] = value

    def failures(self):
        return [c for c in self.checks if c[1] == "error" and not c[2]]
