from dataclasses import dataclass, field
from typing import Optional
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM

def merge_peft_adapter(
    model_name: str = "./lora_output",  # Đường dẫn đến adapter LoRA
    output_name: Optional[str] = None  # Đường dẫn lưu mô hình gộp
) -> None:
    """
    Gộp adapter LoRA vào mô hình gốc và lưu mô hình đã gộp.
    
    Args:
        model_name: Đường dẫn đến thư mục chứa adapter LoRA (args.output_path).
        output_name: Đường dẫn lưu mô hình gộp (args.rl_base_model), mặc định là model_name + "-adapter-merged".
    """
    # Tải cấu hình LoRA từ thư mục
    peft_config = PeftConfig.from_pretrained(model_name)
    print(f"Cấu hình LoRA: {peft_config}")

    # Tải mô hình gốc từ base_model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.float16,  # FP16 để tiết kiệm VRAM
        low_cpu_mem_usage=True,  # Giảm sử dụng RAM
        trust_remote_code=True,  # Cho mô hình DeepSeek
        device_map="auto"  # Phân bổ GPU tự động
    )

    # Tải adapter LoRA và áp dụng vào mô hình
    model = PeftModel.from_pretrained(model, model_name)
    model.eval()  # Chuyển sang chế độ đánh giá

    # Gộp adapter LoRA vào mô hình gốc
    model = model.merge_and_unload()

    # Xác định đường dẫn lưu
    if output_name is None:
        output_name = f"{model_name}-adapter-merged"
    print(f"Lưu mô hình gộp tại: {output_name}")

    # Lưu mô hình đã gộp
    model.save_pretrained(output_name)