from scripts.recognition.benchmark_easyocr import main as run_easyocr
from scripts.recognition.benchmark_trba import main as run_trba
from scripts.recognition.benchmark_trocr import main as run_trocr
from scripts.recognition.summarize_results import write_summary


def main():
    run_trba()
    run_easyocr()
    run_trocr()
    write_summary()
    print("\nAll enabled recognition benchmarks completed.")


if __name__ == "__main__":
    main()
