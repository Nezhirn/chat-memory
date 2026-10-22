from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("usage: python -m app.ingest {from-db,pdf,web,embed} [options]")
        return 0

    command, rest = args[0], args[1:]
    if command == "from-db":
        from app.ingest.from_db import main as command_main
    elif command == "pdf":
        from app.ingest.pdf import main as command_main
    elif command == "web":
        from app.ingest.web import main as command_main
    elif command == "embed":
        from app.ingest.embed import main as command_main
    else:
        print(f"unknown ingest command: {command}", file=sys.stderr)
        print("usage: python -m app.ingest {from-db,pdf,web,embed} [options]")
        return 2
    return command_main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
