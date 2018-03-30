import os


def find_next_run_dir(base_path):
    pattern = os.path.join(base_path, 'run-{}')
    i = 0
    while os.path.exists(pattern.format(i)):
        i += 1
    run_path = pattern.format(i)
    os.makedirs(run_path)
    return run_path
