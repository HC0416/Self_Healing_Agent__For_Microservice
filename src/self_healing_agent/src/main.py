import logging
import threading
import os, time
import random

from common.text_strings import Path_Strings, Style_Strings, Url_Strings
from faults.fault_injector import Faults
from run_auto_query import execute_TT_Pipeline

from train_ticket_auto_query_master.queries import Query
from train_ticket_auto_query_master.scenarios import *






def inject_random_fault(f=None):
    fault_mapping = {
        "F1": Faults.inject_f1,
        "F2": Faults.inject_f2,
        "F3": Faults.inject_f3,
        "F4": Faults.inject_f4,
        "F7": Faults.inject_f7,
        # "F22": Faults.inject_f22,
    }
    
    f = random.choice(list(fault_mapping.keys()))
    
    fault_mapping[f]()
    return f

def background_fault():
    def fault_thread():
        try:
            fault = inject_random_fault()
            print(f"Fault {fault} injected")
        except Exception as e:
            print(f"Fault injection failed: {e}")
    
    thread = threading.Thread(target=fault_thread, daemon=True)
    thread.start()
    time.sleep(4)  

# ── MAIN ──────────────────────────────────────────────
def run():
    
    
    print(Style_Strings.equals_divider_with_start_newline)
    print("FAULT DETECTION DEMO")
    print(Style_Strings.equals_divider_with_end_newline)


    print("[INFO] Cleaning Previous Runs..." )
    Faults.reset()
    print("Done")

    # ── Step 1: Normal baseline ────────────────────────────
    q = Query(Url_Strings.BASE_URL)
    
    print("\n[1/3] Normal baseline - sending login request...")
    try:
        q.login()
        if not q.login():
            logging.fatal('login failed')
        print("[OK] Login successful: 200")
    except Exception as e:
        print(f"[ERROR] Login failed: {e}")
        
    print("Waiting 5s for UI to detect...")
    time.sleep(5)


    # ── Step 2: Inject fault ───────────────────────────────
    print("\n[2/3] Injecting Random Fault...")
    
    # Inject Fault
    background_fault()

    # ── Step 3: Test ──────────────────
    print("\n[3/3] Testing...")

    # Test Logic

    if not q.login():
        logging.fatal('login failed')

    execute_TT_Pipeline()

    Faults.reset()
    print(Style_Strings.equals_divider)
    print("[OK] Run finished")
    print(Style_Strings.equals_divider)








if __name__ == "__main__":
    # ── CONFIG ─────────────────────────────────────────────────
    GLOVE_PATH  = Path_Strings.GLOVE_PATH
    MODELS_PATH = Path_Strings.MODELS_PATH
    MODEL_PATH  = os.path.join(MODELS_PATH, 'results', 'M4_ggnn_normal_only.pt')

    
    run()
    