"""
Example: Multimodal inference (text + image) with the airlock SDK

This script sends a text prompt and a local image to a Phi-4 Multimodal model
running in a container named "phi-mm" using the airlock inference API.

🖼️ Includes a base64-encoded image ('test.png')
💬 Sends a user prompt asking for a description
📡 Prints the assistant's multimodal reply

Runs fully offline, no external API calls.
"""

import base64
import io
from PIL import Image, ImageDraw, ImageFont
from fifo_tool_airlock_model_env.common.models import GenerationParameters, Message, Model, Role
from fifo_tool_airlock_model_env.sdk.client_sdk import call_airlock_model_server

# Read and encode the image as base64

# Read from a file
# with open("test.png", "rb") as f:
#     b64_image = base64.b64encode(f.read()).decode("utf-8")

# NOTE: Pillow's type stubs are incomplete (especially ImageDraw & ImageFont)
#       Pylance strict mode requires `# type: ignore` on certain lines below.

# or directly generate a base64-encoded test image with a square and the word "HELLO"
def make_test_image_b64() -> str:
    img = Image.new("RGB", (640, 640), "white")
    draw = ImageDraw.Draw(img) # type: ignore
    draw.rectangle([40, 40, 600, 600], outline="black", width=4) # type: ignore

    try:
        font = ImageFont.truetype("arial.ttf", 64) # type: ignore
    except IOError:
        font = ImageFont.load_default()

    # Get text bounding box to center it
    bbox = draw.textbbox((0, 0), "HELLO", font=font) # type: ignore
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_position = ((640 - text_w) // 2, (640 - text_h) // 2)
    draw.text(text_position, "HELLO", fill="black", font=font) # type: ignore

    buf = io.BytesIO()
    img.save(buf, format="PNG") # type: ignore
    return base64.b64encode(buf.getvalue()).decode("utf-8")

b64_image = make_test_image_b64()

# Build the prompt
system_prompt = "You are a vision assistant."
user_prompt = "Describe the content of the following image : <|image_1|>"

# Call the multimodal model via the SDK
answer = call_airlock_model_server(
    model=Model.Phi4MultimodalInstruct,   # The model enum for multimodal Phi-4
    adapter=None,
    messages=[
        Message(role=Role.system, content=system_prompt),
        Message(role=Role.user, content=user_prompt)
    ],
    images=[b64_image],                   # The base64-encoded image list
    parameters=GenerationParameters(
        max_new_tokens=1024,
        do_sample=False
    ),
    container_name="phi"                  # Name of the container serving the multimodal model
)

print( "🖼️ Image: test.png")
print(f"💬 User: {user_prompt}")
print(f"🤖 Assistant: {answer}")
