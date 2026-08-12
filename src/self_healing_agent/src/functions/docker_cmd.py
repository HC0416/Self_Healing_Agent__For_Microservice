#!/usr/bin/env python
import subprocess
import sys
import docker
import requests
from requests.exceptions import RequestException


client = docker.from_env()

def get_list():
    try:
        containers = client.containers.list()
        #for container in containers:
            #print(container.name)
        return [container.name for container in containers]
    except docker.errors.APIError as e:
        print(f"Docker API error: {e}")
        return []

def run_cmd(cmd):
    try:
        subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"Command [{cmd}] Succeeded.")
    except Exception:
        print(f"Command [{cmd}] Failed.")


def start_container(container_image, container_name=None, container_volumes=None, container_command=None, container_remove=None):
    try:
        container = client.containers.run(container_image, name=container_name, volumes=container_volumes, command=container_command, remove=container_remove, detach=True)
        return container.logs().decode('utf-8')
    except docker.errors.ContainerError as e:
        print(f"Container error: {e}")
    except docker.errors.ImageNotFound as e:
        print(f"Image not found: {e}")
    except docker.errors.APIError as e:
        print(f"Docker API error: {e}")
        
def pause_container(container_id):
    try:
        container = client.containers.get(container_id)
        container.pause()
        print(f"Container {container_id} paused successfully.")
        return True
    except docker.errors.NotFound:
        print(f"Container {container_id} not found.")
    except docker.errors.APIError as e:
        print(f"Docker API error: {e}")

def unpause_container(container_id):
    try:
        container = client.containers.get(container_id)
        container.unpause()
        print(f"Container {container_id} unpaused successfully.")
        return True
    except docker.errors.NotFound:
        print(f"Container {container_id} not found.")
    except docker.errors.APIError as e:
        print(f"Docker API error: {e}")

def delete_container(container_id):
    try:
        container = client.containers.get(container_id)
        container.remove(force=True)
        print(f"Container {container_id} removed successfully.")
    except docker.errors.NotFound:
        print(f"Container {container_id} not found.")
    except docker.errors.APIError as e:
        print(f"Docker API error: {e}")

def restart_container(container_id):
    try:
        container = client.containers.get(container_id)
        container.restart()
        print(f"Container {container_id} restarted successfully.")
    except docker.errors.NotFound:
        print(f"Container {container_id} not found.")
    except docker.errors.APIError as e:
        print(f"Docker API error: {e}")

def update_memory(container_id, new_memory_limit):
    try:
        container = client.containers.get(container_id)
        container.update(mem_limit=new_memory_limit, memswap_limit=new_memory_limit)
        print(f"Container {container_id} memory limit updated to {new_memory_limit}.")
    except docker.errors.NotFound:
        print(f"Container {container_id} not found.")
    except docker.errors.APIError as e:
        print(f"Docker API error: {e}")

def rollback_container(container_id, older_image_tag):
    try:
        container = client.containers.get(container_id)
        container.stop()
        container.remove()
        new_container = client.containers.run(older_image_tag, detach=True)
        print(f"Container {container_id} rolled back to image {older_image_tag}.")
        return new_container.id
    except docker.errors.NotFound:
        print(f"Container {container_id} not found.")
    except docker.errors.APIError as e:
        print(f"Docker API error: {e}")

def get_container_id(container_id):
    alternate = ["redis"]
    if container_id in alternate:
        id = f"train-ticket-{container_id}-1"
        return id
    if container_id.startswith("ts"):
        id = f"train-ticket-{container_id}-1"
        return id
    return container_id

"""
breakers = {}
def circuit_break(service_id, func, *args, fail_max=3, reset_timeout=15, **kwargs):
    
    service_name = get_container_id(service_id)
    
    if service_name not in breakers:
        breakers[service_name] = pybreaker.CircuitBreaker(
            fail_max=fail_max,
            reset_timeout=reset_timeout
        )
    
    breaker = breakers[service_name]
    
    try:
        return breaker.call(func, *args, **kwargs)
    except pybreaker.CircuitBreakerError:
        print(f"Circuit OPEN for '{service_name}'. Call blocked to prevent cascading failure.")
        return None
        
    except Exception as e:
        print(f"Call to [{service_name}] failed: {e}")
        raise e
"""



if __name__ == "__main__":
    #pass
    #client.containers.run("ubuntu", "echo hello world")
    #print(output)
    op = client.containers.list()
    #print(op)
    for container in op:
        print(container.name)
    #test = get_list()
    #c = test[1]
    container = "redis"
    print(get_container_id(container))