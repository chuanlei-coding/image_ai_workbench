# GLM-Image Web UI

一个基于 FastAPI 的 Web UI 应用，为 [zai-org/GLM-Image](https://huggingface.co/zai-org/GLM-Image) 大模型提供可视化的图像生成界面。支持文本生成图像（Text-to-Image）和图像生成图像（Image-to-Image）两种功能。

## 功能特性

- 🎨 **文本生成图像（Text-to-Image）**：根据用户输入的提示词生成图像
- 🖼️ **图像生成图像（Image-to-Image）**：支持上传多张图像，结合提示词生成新图像
- 🎯 **实时预览**：生成的图像可以实时预览
- 💾 **图像下载**：支持下载生成的图像
- ⚙️ **参数可调**：可调整图像尺寸、推理步数、引导强度、随机种子等参数
- 🚀 **高性能**：模型在应用启动时预加载，避免首次请求延迟
- 🎨 **现代化 UI**：美观的响应式 Web 界面

## 环境要求

- Python 3.8+
- CUDA 支持的 GPU（推荐）
- 足够的显存（建议 8GB+）

## 安装步骤

1. **克隆仓库**
   ```bash
   git clone git@github.com:chuanlei-coding/image_ai_workbench.git
   cd image_ai_workbench
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **运行应用**
   ```bash
   python glm_image_ui.py
   ```

4. **访问 Web UI**
   
   应用启动后，在浏览器中访问：`http://localhost:7860`

   > 注意：首次启动时，模型会自动从 Hugging Face 下载，可能需要一些时间。模型加载完成后即可使用。

## 使用方法

### 文本生成图像（Text-to-Image）

1. 在 "Text-to-Image" 标签页中
2. 输入提示词（Prompt）
3. 调整参数（可选）：
   - **Height/Width**：图像尺寸（默认 1024x1152）
   - **Inference Steps**：推理步数（默认 50，范围 10-100）
   - **Guidance Scale**：引导强度（默认 1.5，范围 1.0-10.0）
   - **Seed**：随机种子（默认 42，使用 -1 表示随机）
4. 点击 "Generate" 按钮
5. 等待生成完成后，可以预览和下载图像

### 图像生成图像（Image-to-Image）

1. 在 "Image-to-Image" 标签页中
2. 上传一张或多张图像（支持多选）
3. 输入提示词（Prompt）
4. 调整参数（可选）：
   - **Height/Width**：输出图像尺寸（默认 1056x1024）
   - **Inference Steps**：推理步数（默认 50）
   - **Guidance Scale**：引导强度（默认 1.5）
   - **Seed**：随机种子（默认 42）
5. 点击 "Generate" 按钮
6. 等待生成完成后，可以预览和下载图像

## API 文档

### 文本生成图像 API

**端点**：`POST /api/text-to-image`

**请求体**（JSON）：
```json
{
  "prompt": "your prompt text",
  "height": 1024,
  "width": 1152,
  "num_inference_steps": 50,
  "guidance_scale": 1.5,
  "seed": 42
}
```

**响应**：
```json
{
  "image": "data:image/png;base64,...",
  "status": "success"
}
```

### 图像生成图像 API

**端点**：`POST /api/image-to-image`

**请求**（Form Data）：
- `files`: 图像文件（支持多文件）
- `prompt`: 提示词
- `height`: 图像高度（默认 1056）
- `width`: 图像宽度（默认 1024）
- `num_inference_steps`: 推理步数（默认 50）
- `guidance_scale`: 引导强度（默认 1.5）
- `seed`: 随机种子（默认 42）

**响应**：
```json
{
  "image": "data:image/png;base64,...",
  "status": "success"
}
```

## 技术栈

- **后端框架**：FastAPI
- **Web 服务器**：Uvicorn
- **深度学习框架**：PyTorch
- **模型库**：Diffusers
- **图像处理**：Pillow (PIL)
- **前端**：原生 HTML/CSS/JavaScript

## 项目结构

```
image_ai_workbench/
├── glm_image_ui.py      # 主应用文件（包含 FastAPI 后端和前端 HTML）
├── requirements.txt     # Python 依赖列表
└── README.md           # 项目说明文档
```

## 注意事项

1. **模型下载**：首次运行时，模型会自动从 Hugging Face 下载，需要网络连接
2. **显存要求**：GLM-Image 模型需要较大的显存，建议使用 GPU 运行
3. **生成时间**：图像生成时间取决于参数设置和硬件性能，通常需要几秒到几十秒
4. **并发请求**：当前实现为单线程处理，如需支持并发，建议使用负载均衡器

## 开发说明

### 修改端口

在 `glm_image_ui.py` 文件末尾修改：
```python
uvicorn.run(app, host="0.0.0.0", port=7860)  # 修改端口号
```

### 修改模型加载参数

在 `load_model()` 函数中可以调整模型加载参数：
```python
pipe = GlmImagePipeline.from_pretrained(
    "zai-org/GLM-Image",
    device_map="cuda"  # 可以添加 torch_dtype=torch.bfloat16 等参数
)
```

## 许可证

本项目基于 MIT 许可证开源。

## 致谢

- [zai-org/GLM-Image](https://huggingface.co/zai-org/GLM-Image) - 图像生成模型
- [FastAPI](https://fastapi.tiangolo.com/) - 现代 Web 框架
- [Diffusers](https://github.com/huggingface/diffusers) - 扩散模型库

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系。
