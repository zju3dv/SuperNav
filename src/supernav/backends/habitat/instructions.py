"""Instructions for the Habitat execution environment."""


def habitat_instructions(agent_backend: str) -> str:
    qualifier = "habitat-gs " if agent_backend == "kimi" else ""
    return (
        "You are running a controlled Habitat-GS benchmark episode. "
        f"Use only the {qualifier}MCP tools exposed for scene perception and action, "
        "and follow the user prompt exactly.\n"
    )
