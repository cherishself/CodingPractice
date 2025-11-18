1. **确认 CUDA 环境**  
   - 在命令行输入：  
     ```bash
     nvcc -V
     ```  
     若能显示 CUDA 版本信息，说明 CUDA 已安装成功。

2. **下载 cuDNN**  
   - 前往 [NVIDIA cuDNN 下载页面](https://developer.nvidia.com/rdp/cudnn-download)（需要注册账号）。  
   - 根据你的 CUDA 版本选择对应的 cuDNN 版本。例如 CUDA 12.1 对应 cuDNN 8.x。

3. **解压并复制文件**  
   下载并解压后，将以下文件复制到 CUDA 安装目录：  
   - `bin` → 复制到 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\bin`  
   - `include` → 复制到 `...\include`  
   - `lib` → 复制到 `...\lib\x64`

4. **配置环境变量**  
   - 确认 `CUDA_PATH` 已指向正确版本。  
   - 确认 `PATH` 中包含 `...\bin` 和 `...\libnvvp`。

---

## ✅ 验证 cuDNN 是否安装成功

1. **运行 NVIDIA Demo 工具**  
   在 CUDA 安装目录下的 `extras/demo_suite` 中运行：  
   - `deviceQuery.exe` → 若输出 GPU 信息，说明 CUDA 正常。  
   - `bandwidthTest.exe` → 测试显存带宽。注意：这只能验证 CUDA，不代表 cuDNN。

2. **使用 PyTorch 验证 cuDNN**  
   在已安装 PyTorch 的环境中运行：  
   ```python
   import torch
   print(torch.backends.cudnn.version())
   print(torch.backends.cudnn.is_available())
   ```  
   - 若返回 cuDNN 版本号（比如 `8905`）且 `True`，说明 cuDNN 已正确安装。
