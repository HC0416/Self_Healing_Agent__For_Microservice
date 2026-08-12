class Message_Strings:
    
    # Possible messages for different scenarios
    error_message = "An error occurred."
    success_message = "Operation completed successfully."
    warning_message = "Warning!"
    assessing_message = "Assessing... Please Wait..."

    
    # Execution messages for different actions
    
    execution_message_restart = "Restarting Container..."
    execution_message_scale_up = "Scaling the container's memory limit..."
    execution_message_rollback = "Rolling back the container to the previous image..."

class Style_Strings:
    divider = "-" * 70
    equals_divider_with_start_newline = "\n" + "=" * 70
    equals_divider = "=" * 70
    equals_divider_with_end_newline = "=" * 70 + "\n"
    equals_divider_with_start_and_end_newline = "\n" + ("=" * 70) + "\n"

class Model_Strings:
    
    llm_3_model = "Qwen/Qwen2.5-Coder-3B-Instruct"
    llm_7_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
    llm_14_model = "Qwen/Qwen2.5-Coder-14B-Instruct"
    
    model_role = "You are an expert site reliability engineer for a microservice system running the 'Train Ticket' benchmark microservice."

    structured_prompt_message = "Explain the Train Ticket Github Repo?"

    test_prompt_message = "If I was to train a self healing model on an microservice so that it can take a stack trace of an anomaly and return reasoning on what to do. What hugging-face or otherwise models and steps would I take. I use python and the project uses Docker."

    model_prompt_requirements = f"""

        Without using markdown, please provide:
        1. Root Cause Analysis
        2. Why this service is likely responsible
        3. Files/classes that should be inspected
        4. Recommended code changes
        5. Runtime recovery actions
        6. Confidence score
        """


class Path_Strings:
    MODELS_PATH = r'C:\Users\eamon\Documents\code\projects\2026-mcm-cullime2-hadrian2\src\self_healing_agent\src\models'
    GLOVE_PATH  = r"C:\Users\eamon\Downloads\glove.6B\glove.6B.300d.txt"
    A3_PATH = r"C:\Users\eamon\Documents\code\projects\2026-mcm-cullime2-hadrian2\src\self_healing_agent\src\rl_models\results\A3_reinforce_pretrain.pt"
    
class Url_Strings:
    BASE_URL = "http://localhost:8080"
    SW_GRAPHQL  = "http://localhost:12800/graphql"
    AUTH_LOGIN  = "http://localhost:12340/api/v1/users/login"

class Admin_API:
    ADMIN_BEARER_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhZG1pbiIsInJvbGVzIjpbIlJPTEVfQURNSU4iXSwiaWQiOiJlOWQyYWE1OC01YTNlLTQ0NjMtOWIxNy0xOGY5MjczNGM4YTciLCJpYXQiOjE3ODM5ODI5MzQsImV4cCI6MTc4Mzk4NjUzNH0.xtFmJCmWfqzABdIMyiYaDpxt8H7jzrjTakEopTEXgBk"
    ADMIN_API_URL = "http://localhost:16112/api/v1/adminorderservice/adminorder"
    ADMIN_USER_ID = "4d2a46c7-71cb-4cf1-b5bb-b68406d9da6f"
    ADMIN_ORDER_ID = "a36d97b5-9e81-4093-b2a2-2f4cd4f98c70"
    