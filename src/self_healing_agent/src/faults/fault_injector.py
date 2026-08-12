import requests
import time
from functions.docker_cmd import *


class Faults:
    def __init__(self):
        pass
    
    # INDIVIDUAL FAULT INJECTIONS

    def inject_f1():
        """F1: Asynchronous task delay in ticket-cancellation logic."""
        name = "ts-auth-service"
        container = get_container_id(name)
        duration = 30
        print(f"Injecting F1: Freezing {container} for {duration}s...")
        
        pause = pause_container(container)
        if pause:
            time.sleep(duration)
            unpause_container(container)
            print(f"Unfroze {container}. Race condition fault triggered.")
        else:
            print(f"Failed to pause {container}. Verify the container is running.")

    def inject_f2():
        """F2: Scrambled UI response order due to ticket reservation delay."""
        name = "ts-order-service"
        container = get_container_id(name)
        duration = 30
        print(f"Injecting F2: Freezing {container} for {duration}s...")
        
        pause = pause_container(container)
        if pause:
            time.sleep(duration)
            unpause_container(container)
            print(f"Unfroze {container}.")
        else:
            print(f"Failed to pause {container}.")

    def inject_f3():
        """F3: JVM vs Docker Memory Misconfiguration."""
        name = "ts-order-service"
        memory = "128M"
        container = get_container_id(name)
        print(f"Injecting F3: Constraining {container} memory runtime to {memory}...")
        
        update_memory(container, memory)
        print(f"Memory constraints applied to {container}. Send heavy traffic to trigger crash.")

    def inject_f4():
        print("Run Clumsy to Simulate Packet Loss.")

    def inject_f7():
        """F7: Third-party service payment timeout emulation."""
        name = "ts-payment-service"
        container = get_container_id(name)
        duration = 30
        print(f"Injecting F7: Freezing {container} for {duration}s...")
        
        pause = pause_container(container)
        if pause:
            time.sleep(duration)
            unpause_container(container)
            print(f"Unfroze {container}. Timeout exception successfully triggered.")
        else:
            print(f"Failed to pause {container}.")

    # ==========================================================

    def inject_f10():
        payload = {
            "userId": "123",
            "scenario": "edge_case"
        }
        url = "http://localhost:8080//api/v1/stationservice/stations/namelist"

        for i in range(0,10): 
            requests.post(url, json=payload)
            i+=1

        print("F10 injected - wrong API used")
        return

    def inject_f15():
        """F15: Nginx Max Body Size Limitation (200 Bytes)."""
        print("Injecting F15: Overwriting local Nginx rules to cap request payloads at 200 bytes...")
        
        nginx_cmd = ""
        res = run_cmd(nginx_cmd)
        print("Nginx successfully reloaded with payload blockades active.")

    def inject_f22():
        """F22: Malformed SQL schema (Column missing error)."""
        print("Injecting F22: Breaking Auth database column structure...")
        
        sql = "USE `ts-auth-mysql`; ALTER TABLE auth_user CHANGE COLUMN user_id wrong_column_name varchar(36);"
        cmd = f'docker exec -i ts-auth-mysql mysql -u root -pAbcd1234# -e "{sql}"'
        run_cmd(cmd)
        print("Done")


    # SYSTEM CLEANUP & Reset

    def reset():
        
        print("Resetting...")
        
        print("Unpausing any container processes...")
        run_cmd("docker unpause $(docker ps -aq)") 
        
        print("Releasing oder-service resource limits...")
        update_memory("ts-order-service", "1g")
        run_cmd("docker-compose restart") 

# CLI HANDLING ROUTINE


if __name__ == "__main__":
    """
    fault_mapping = {
        "F1": inject_f1,
        "F2": inject_f2,
        "F3": inject_f3,
        "F4": inject_f4,
        "F6": inject_f6,
        "F7": inject_f7,
        #"F8": inject_f8,
        #"F9": inject_f9,
        #"F10": inject_f10,
        #"F15": inject_f15,
        "F22": inject_f22,
        "RESET": reset
    }
    
    
    
    if len(sys.argv) < 2 or sys.argv[1].upper() not in fault_mapping:
        print(f"Fault: [{sys.argv[1].upper()}] not found!")
        sys.exit(1)
        
    target_fault = sys.argv[1].upper()
    
    fault_mapping[target_fault]()
    """