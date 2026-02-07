from __future__ import annotations

import argparse
from typing import Optional

def add_embed_chunks(subparsers: argparse._SubParsersAction):
    p = subparsers.add_parser("embed_chunks", help="Semantic Search CLI")
    p.set_defaults(func=lambda a: cmd_embed_chunks())



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kwsearch",
        description="Keyword Search CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # required=True works on modern Python; fall back for older versions if needed
    subparsers = parser.add_subparsers(
        title="commands", dest="command", required=True
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Each subcommand sets a .func; calling it performs the action
    result = args.func(args)
    return 0 if result is None else int(bool(result))


if __name__ == "__main__":
    raise SystemExit(main())