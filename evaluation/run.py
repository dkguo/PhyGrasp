# run "python -m evaluation.main_multi --top_n 5 -n 10000" multiple times

import os

if __name__ == '__main__':
    total = 10000
    every_time = 800
    start = 2000
    for n in range(start, total + every_time, every_time):
        print(f'n = {n}')
        try:
            os.system(f"python -m evaluation.main_multi --top_n 5 -n {n}")
        except KeyboardInterrupt:
            break

            