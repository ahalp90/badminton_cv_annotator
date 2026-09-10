import torch
from transformers import AutoModelForCausalLM, AutoProcessor

model_path = "yanziang/InternVideo3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto",
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
)






video_path = "videos/test_video.mp4"

fps = 1
min_pixels = 128 * 32 * 32
max_pixels = 128 * 32 * 32

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "fps": fps},
            {"type": "text", "text": "Please describe this video in detail."},
        ],
    }
]

# processor.video_processor.size = {
#     "longest_edge": max_pixels * max_frames,
#     "shortest_edge": min_pixels * min_frames,
# }

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    fps=fps,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

output = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
generated_ids = [o[len(i):] for i, o in zip(inputs.input_ids, output)]
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
