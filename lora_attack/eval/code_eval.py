import faulthandler
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import signal
import time
from typing import List
import os


def execute_code(code):
    try:
        exec(code, {})
    except:
        return False
    return True


def extract_code_from_generation(output: str):
    """
    Produces the prefix of output that ends at the first occurrence of
    a stop word.
    WARNING: the output *must not* include the prompt, which may have stop tokens
    itself.
    """
    stop_words = ["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```"]
    stop_words = []
    min_stop_index = len(output)
    for stop_token in stop_words:
        stop_index = output.find(stop_token)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return output[:min_stop_index]


def run_code_with_timeout(code: str, test: List[str]) -> bool:
    """Run code in a separate process with timeout"""

    def worker():
        reliability_guard()
        try:
            full_code = extract_code_from_generation(code)
            full_code += "\n" + "\n".join(test)
            return execute_code(full_code)
        except Exception:
            return False

    # Create pipe for result communication
    parent_conn, child_conn = multiprocessing.Pipe()

    # Start process
    process = multiprocessing.Process(target=lambda: child_conn.send(worker()))
    process.start()

    try:
        if parent_conn.poll(2.0):  # 2 second timeout
            result = parent_conn.recv()
        else:
            result = False
    except Exception:
        result = False
    finally:
        # Ensure process is terminated
        if process.is_alive():
            process.terminate()
            process.join(timeout=0.1)
            if process.is_alive():
                os.kill(process.pid, signal.SIGKILL)

        # Clean up pipes
        parent_conn.close()
        child_conn.close()

    return result


def process_chunk(chunk: List[tuple[int, tuple[List[str], str]]]) -> List[tuple[int, bool]]:
    """Process a chunk of test cases"""
    results = []
    for idx, (test, code) in chunk:
        result = run_code_with_timeout(code, test)
        results.append((idx, result))
    return results


def run_code_in_process(tests: List[List[str]], codes: List[str]) -> List[bool]:
    """
    Run multiple code tests in parallel with proper process management.
    Maintains max 6 concurrent processes and ensures proper cleanup.
    """
    assert len(tests) == len(codes), "Number of tests and codes must be equal"

    # Prepare work items
    work_items = list(enumerate(zip(tests, codes)))
    chunk_size = max(1, len(work_items) // 6)  # Split work into chunks
    chunks = [work_items[i:i + chunk_size] for i in range(0, len(work_items), chunk_size)]

    results = [False] * len(tests)

    # Use ThreadPoolExecutor to manage process chunks
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]

        # Collect results
        for future in as_completed(futures):
            try:
                chunk_results = future.result()
                for idx, result in chunk_results:
                    results[idx] = result
            except Exception:
                continue

    return results


def reliability_guard(maximum_memory_bytes=None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    """
    if maximum_memory_bytes is not None:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins
    builtins.exit = None
    builtins.quit = None

    import os
    os.environ["OMP_NUM_THREADS"] = "1"

    # Disable potentially harmful functions
    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess
    subprocess.Popen = None

    __builtins__["help"] = None

    import sys
    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None