"""
Confirm whether the --configuration= argument parsing bug is real.

The current code in gunfolds/utils/clingo.py line 63 passes:
    ["--warn=no-atom-undefined", "--configuration=", "crafty", ...]

This test checks whether clingo interprets that as configuration="crafty"
or ignores it (defaulting to "auto") by comparing the active configuration
reported by the Control object in both the buggy and correct formats.

Run:  python test_clingo_config_bug.py
"""

import clingo

ASP_PROGRAM = "a(1). {b(X)} :- a(X). #show b/1."


def get_active_config(args, label):
    """Create a Control with given args and report the active configuration."""
    ctrl = clingo.Control(args)
    config = ctrl.configuration

    solver_conf = str(config.configuration)

    ctrl.add("base", [], ASP_PROGRAM)
    ctrl.ground([("base", [])])
    models = []
    with ctrl.solve(yield_=True) as handle:
        for m in handle:
            models.append([str(a) for a in m.symbols(shown=True)])

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  args passed:           {args}")
    print(f"  configuration value:   {solver_conf!r}")
    print(f"  models found:          {len(models)}")
    return solver_conf


print("Clingo version:", clingo.__version__)
print()

buggy = get_active_config(
    ["--warn=no-atom-undefined", "--configuration=", "crafty",
     "-t", "1,split", "-n", "0"],
    "BUGGY FORMAT  (current gunfolds code)"
)

correct = get_active_config(
    ["--warn=no-atom-undefined", "--configuration=crafty",
     "-t", "1,split", "-n", "0"],
    "CORRECT FORMAT  (proposed fix)"
)

default = get_active_config(
    ["--warn=no-atom-undefined",
     "-t", "1,split", "-n", "0"],
    "NO --configuration FLAG  (clingo default)"
)

explicit_auto = get_active_config(
    ["--warn=no-atom-undefined", "--configuration=auto",
     "-t", "1,split", "-n", "0"],
    "EXPLICIT --configuration=auto"
)

print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
print(f"  Buggy format config:    {buggy!r}")
print(f"  Correct format config:  {correct!r}")
print(f"  No flag (default):      {default!r}")
print(f"  Explicit auto:          {explicit_auto!r}")
print()

if buggy == correct:
    print("  RESULT: Bug NOT confirmed -- clingo handles the split args OK.")
    print("          (Both formats produce the same configuration.)")
else:
    print("  RESULT: BUG CONFIRMED!")
    print(f"          Buggy format gives config={buggy!r}")
    print(f"          Correct format gives config={correct!r}")
    if buggy == default:
        print(f"          Buggy format falls back to the default ({default!r}),")
        print(f"          meaning --configuration=crafty was NEVER applied.")
