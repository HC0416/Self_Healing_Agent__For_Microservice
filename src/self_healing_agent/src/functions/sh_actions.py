#!/usr/bin/env python
from self_healing_code.functions.docker_cmd import *

class SHActions:
    def __init__(self):
        pass
    
    def execute_action(action_ID, container_id, image_name=None, command=None, new_memory_limit=None):
        match action_ID:
            case 0:
                return restart_container(container_id)
            case 1:
                return update_memory(container_id, new_memory_limit)
            case 2:
                return None # circuit_break(container_id, image_name, command)
            case 3:
                return None # reroute_container(container_id, image_name, command)
            case 4:
                return rollback_container(container_id, image_name)
            case _:
                raise ValueError(f"Invalid action ID: {action_ID}")

