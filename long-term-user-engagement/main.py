from time import time

from data import load_obd
from runner import evaluate_policies, set_plot_style


def main():
    set_plot_style()
    total_start = time()

    feedback = load_obd()

    evaluate_policies(feedback=feedback)

    print(f"\n[TOTAL] {time() - total_start:.2f} sec")


if __name__ == "__main__":
    main()
