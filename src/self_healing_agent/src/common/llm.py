from common.text_strings import Model_Strings
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLM:
    def __init__(self):
        pass
    

    def load_reasoning_llm(model=Model_Strings.llm_3_model):
        print("Loading LLM...")

        model_name = model
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto"
        )

        print("LLM loaded")
        return llm, tokenizer



    def ask_llm(action, state, score, normal_score, root_cause, service_name, llm, tokenizer):


        prompt = f"""
            Service Name:
            {service_name}
            
            Normal Score:
            {normal_score}
            
            State:
            {state}
            
            Detected anomaly score:
            {score}

            Predicted root cause:
            {root_cause}

            Action Taken
            {action}
            
            """
            
        prompt += Model_Strings.model_prompt_requirements
            
        messages = [
            {
                "role": "system",
                "content": Model_Strings.model_role
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer(
            [text],
            return_tensors="pt"
        ).to(llm.device)

        generated_ids = llm.generate(
            **model_inputs,
            max_new_tokens=1000,
            temperature=0.7,
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response