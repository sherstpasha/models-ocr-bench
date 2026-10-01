import subprocess
import sys

from configs.benchmark_config import BENCHMARKS
from scripts.summarize_results import write_summary


def enabled_modules():
    modules = []
    for config in BENCHMARKS.values():
        if not config.get("run", False):
            continue
        module = f"scripts.{config['script'].stem}"
        if module not in modules:
            modules.append(module)
    return modules


def main():
    modules = enabled_modules()
    if not modules:
        print("No benchmarks enabled in configs/benchmark_config.py")
        return

    for index, module in enumerate(modules, start=1):
        print(f"\n[{index}/{len(modules)}] Running {module}", flush=True)
        subprocess.run([sys.executable, "-m", module], check=True)

    write_summary()
    print("\nAll enabled benchmarks completed.")


if __name__ == "__main__":
    main()
