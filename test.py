# # from transformers import LlamaConfig
# # config = LlamaConfig.from_pretrained("src/llama_recipes/configs/config_micro")
# # print(type(config))  # 应该输出：<class 'transformers.models.llama.configuration_llama.LlamaConfig'>
# import torch
#
#
# def inspect_checkpoint(file_path):
#     print(f"🔍 正在加载检查点: {file_path}")
#     print("⏳ 这可能需要几秒钟，请稍候...\n")
#
#     # 强制使用 CPU 加载，避免占用宝贵的 4070 显存
#     try:
#         checkpoint = torch.load(file_path, map_location='cpu')
#     except Exception as e:
#         print(f"❌ 加载失败: {e}")
#         return
#
#     # 判断它是单纯的 state_dict 还是包含了 optimizer 等信息的综合字典
#     if isinstance(checkpoint, dict):
#         print(f"📁 顶层键值 (Top-level keys): {list(checkpoint.keys())}\n")
#
#         # 提取真正的模型参数字典
#         state_dict = checkpoint.get('model_state_dict', checkpoint)
#
#         print("=" * 70)
#         print(f"📦 模型参数结构 (共 {len(state_dict)} 个张量):")
#         print("=" * 70)
#
#         decoder_keys = []
#         head_keys = []
#         embed_keys = []
#
#         for key, value in state_dict.items():
#             # 获取张量的形状 (如 [32000, 1024])
#             shape_str = str(list(value.shape)) if isinstance(value, torch.Tensor) else str(type(value))
#
#             # 分类收集，方便你一眼看出问题所在
#             if 'decoder' in key:
#                 decoder_keys.append((key, shape_str))
#             if 'embed' in key:
#                 embed_keys.append((key, shape_str))
#             if 'head' in key:
#                 head_keys.append((key, shape_str))
#
#         print("\n🎯 1. 包含 'decoder' 的层:")
#         if not decoder_keys: print("  (空)")
#         for k, s in decoder_keys: print(f"  {k}: {s}")
#
#         print("\n🎯 2. 包含 'embed' 的层 (这里藏着词表信息):")
#         if not embed_keys: print("  (空)")
#         for k, s in embed_keys: print(f"  {k}: {s}")
#
#         print("\n🎯 3. 包含 'head' 的层 (输出层):")
#         if not head_keys: print("  (空)")
#         for k, s in head_keys: print(f"  {k}: {s}")
#
#     else:
#         print(f"⚠️ 这个 .pt 文件不是标准的字典格式，它的类型是: {type(checkpoint)}")
#
#
# if __name__ == "__main__":
#     # 指向你的 309M 权重文件路径
#     ckpt_path = "model_checkpoints/moonbeam_309M.pt"
#     inspect_checkpoint(ckpt_path)

import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))