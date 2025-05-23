"""
Example: Calling a model locally via the airlock SDK

This script sends a structured prompt to the Phi 4 Mini Instruct model
running in a container named "phi" using the local airlock inference setup.

🧠 Starts with a system prompt to set assistant behavior
💬 Sends a user message
📡 Prints the assistant's reply

Runs fully offline with no external API calls
"""

from fifo_tool_airlock_model_env.common.models import GenerationParameters, Message, Model, Role
from fifo_tool_airlock_model_env.sdk.client_sdk import call_airlock_model_server

# Define the system message, this sets the assistant's behavior
system_prompt = "You are a helpful assistant."

# Define the user's input
user = "how are you?"

# Send the structured request to the airlock model server
answer = call_airlock_model_server(
    model=Model.Phi4MiniInstruct,          # The model enum identifier
    adapter=None,                          # Optional: provide an adapter name or None
    messages=[
        Message(role=Role.system, content=system_prompt),
        Message(role=Role.user, content=user)
    ],
    parameters=GenerationParameters(
        max_new_tokens=1024,
        do_sample=False                    # Deterministic output
    ),
    container_name="phi"                   # Name of the container serving the model
)

# Display the prompt and the model's response
print(f"💬 User: {user}")
print(f"🤖 Assistant: {answer}")
