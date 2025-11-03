"""Utils for the app."""

import base64
import io
import mimetypes
import typing
from google.genai import types
import gradio as gr
from PIL import Image

def get_part_from_file(file):
    """Help function to get the part from a file."""
    guessed_type = mimetypes.guess_type(file)
    if guessed_type:
        mime_type = guessed_type[0]
    else:
        mime_type = "application/octet-stream"
    with open(file, "rb") as f:
        data = f.read()
        return types.Part.from_bytes(
            data=data,
            mime_type=mime_type,
        )

def get_bytes_from_image(image: Image.Image, mime_type: str = "PNG") -> bytes:
    """Converts a PIL Image object to bytes in the specified format."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=mime_type)
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def get_parts_from_message(
    message: typing.Union[str, tuple[str, ...], dict[str, str], gr.Image],
):
    """Help function to get the parts from a message."""

    parts = []
    if isinstance(message, dict):
        if "text" in message and message["text"]:
            # FIX: Use from_bytes instead of from_text to avoid argument error
            parts.append(types.Part.from_bytes(
                data=message["text"].encode('utf-8'),
                mime_type="text/plain"
            ))
        if "files" in message:
            for file in message["files"]:
                parts.append(get_part_from_file(file))
    elif isinstance(message, str):
        if message:
            # FIX: Use from_bytes instead of from_text to avoid argument error
            parts.append(types.Part.from_bytes(
                data=message.encode('utf-8'),
                mime_type="text/plain"
            ))
    elif isinstance(message, gr.Image):
        if message.type == "pil":
            bytes_data = get_bytes_from_image(message.value)
            parts.append(
                types.Part.from_bytes(data=bytes_data, mime_type=message.format)
            )
        elif message.type == "filepath":
            parts.append(get_part_from_file(message.value))
    else:
        for part in list(message):
            if part.startswith("/tmp/gradio"):
                parts.append(get_part_from_file(part))
            elif part:
                # FIX: Use from_bytes instead of from_text to avoid argument error
                parts.append(types.Part.from_bytes(
                    data=part.encode('utf-8'),
                    mime_type="text/plain"
                ))

    # To avoid error when sending empty message.
    if not parts:
        parts.append(types.Part.from_bytes(
            data=" ".encode('utf-8'),
            mime_type="text/plain"
        ))
    return parts
