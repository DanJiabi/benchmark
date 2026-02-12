# AGENTS.md

本文档包含在此仓库工作的代理的指南和命令。

## 环境设置

- 平台：macOS（Apple Silicon），兼容 Linux
- Python 环境：使用 mamba 管理虚拟环境（位于 ~/miniforge3）
- 默认环境名称：与项目文件夹名称相同（"benchmark"）
- 环境配置：通过 `environment.yml` 管理
- AI 硬件加速优先级：cuda → mps → cpu

## 构建/测试命令

```bash
# 创建或更新虚拟环境
mamba env create -f environment.yml --force
mamba env update -f environment.yml --prune

# 激活环境
conda activate benchmark

# 运行测试（当 pytest 可用时）
pytest -v                              # 运行所有测试（详细输出）
pytest tests/test_file.py              # 运行单个测试文件
pytest tests/test_file.py::test_function  # 运行单个测试函数
pytest tests/ -k "pattern"             # 运行匹配模式的测试
pytest tests/ -m "marker"              # 运行带特定标记的测试
pytest tests/ -x                        # 遇到首次失败时停止
pytest tests/ --cov=src                 # 运行测试并生成覆盖率报告
pytest tests/ --durations=10            # 显示 10 个最慢的测试

# 代码质量（当可用时）
ruff check .                            # 检查代码
ruff check --fix .                      # 自动修复 lint 问题
ruff format .                           # 格式化代码
ruff check --select I                   # 检查导入顺序
mypy .                                  # 类型检查
mypy --strict .                         # 严格类型检查

# 安装依赖
mamba install -y package_name
pip install -r requirements.txt
pip install -e .                        # 以可编辑模式安装包
```

## 代码风格指南

### 导入
- 使用 `isort` 或遵循 Python PEP 8 导入顺序（stdlib → third-party → local）
- 在各组之间使用空行分隔导入
- 优先使用绝对导入
- 避免通配符导入（`from module import *`）

### 格式化
- 使用 `ruff` 进行代码格式化（符合 PEP 8）
- 行长度：88 个字符（ruff 默认值）
- 使用 4 个空格缩进，不使用制表符

### 类型
- 为函数签名和变量使用类型提示
- 优先使用类型别名而非复杂的内联类型
- 使用 `Optional[T]` 或 `T | None` 表示可选值
- 必要时为第三方库添加类型存根

### 命名约定
- 函数/变量：`snake_case`
- 类：`PascalCase`
- 常量：`UPPER_SNAKE_CASE`
- 私有成员：`_leading_underscore`

### 错误处理
- 使用特定异常，避免广泛的 `except Exception:`
- 在错误消息中包含上下文
- 使用 logging 而非 print 进行调试
- 在 `finally` 块或上下文管理器中清理资源

## 测试

- 在 `tests/` 目录中编写测试
- 使用描述性测试名称（`test_feature_scenario`）
- 保持测试隔离和确定性
- 使用 fixtures 进行设置/拆卸
- 为长时间运行的测试添加超时

## 临时文件

- 将测试/验证脚本放在项目根目录的 `.temp/` 目录中
- 定期清理 `.temp/` 或将其添加到 `.gitignore`

## 硬件加速

对于深度学习项目：
- 首先检查 CUDA 可用性
- 回退到 MPS（Apple Silicon）
- 最终回退到 CPU
- 使用设备检测工具而非硬编码

## 提交指南

- 遵循仓库中现有的提交消息风格
- 提交前运行 linting 和测试
- 永不提交密钥、凭据或环境文件
- 如适用，使用 conventional commits 格式

## 调试

- 使用 Python 内置的 `pdb` 或 `ipdb` 进行调试
- 使用 `breakpoint()` 或 `pdb.set_trace()` 设置断点
- 使用适当的日志级别（DEBUG、INFO、WARNING、ERROR）
- 避免使用 `print()` 语句 - 改用 logging
- 对于长时间运行的操作，添加进度指示器

## 文档

- 为所有公共函数和类使用 Google 或 NumPy 风格编写文档字符串
- 保持文档字符串与代码更改同步
- 在类型提示能增加清晰度时在文档字符串中包含它们
- 记录边缘情况和错误条件
- 谨慎使用内联注释 - 优先使用自解释代码
- 添加主要功能时更新 README.md

## 代码组织

- 保持模块专注，尽可能在 300 行以内
- 使用 `__init__.py` 暴露清晰的公共 API
- 分离关注点：模型、视图、控制器应保持独立
- 使用依赖注入提高可测试性
- 优先组合而非继承
- 如适用，遵循 SOLID 原则

## 性能

- 优化前先分析代码（使用 `cProfile` 或 `timeit`）
- 考虑算法复杂度（时间和空间）
- 对大型数据集使用生成器
- 适当时缓存昂贵操作
- 避免过早优化
- 对于 ML/深度学习：使用批处理、向量化

## 安全

- 验证所有用户输入
- 使用参数化查询防止 SQL 注入
- 永不记录敏感数据（密码、令牌、API 密钥）
- 使用环境变量管理配置机密
- 定期保持依赖项更新
- 对外部网络调用使用 HTTPS
- 清理文件路径以防止路径遍历

## Git 工作流程

- 从 main/master 创建功能分支
- 编写描述性分支名称（例如 `feature/add-authentication`）
- 推送前拉取最新更改
- 使用描述性 PR 标题和描述
- 合并前请求代码审查
- 先在本地解决合并冲突

## 常见运行问题

### Apple Silicon MPS 设备兼容性

**问题**: `torchvision::nms` 操作不支持 MPS 后端

**解决方案**:
```bash
# 方式 1：使用启动脚本（推荐）
./run_benchmark.sh --config config.yaml --model yolov8n

# 方式 2：手动设置环境变量
export PYTORCH_ENABLE_MPS_FALLBACK=1
python benchmark.py --config config.yaml --model yolov8n
```

### 相对导入错误

**问题**: `ImportError: attempted relative import beyond top-level package`

**解决方案**: 从项目根目录运行，使用绝对导入路径
```bash
cd /Users/danjiabi/github/benchmark
python benchmark.py --config config.yaml --model yolov8n
```

### --model all 模式错误

**问题**: `--model all` 显示"计划测试 0 个模型"并在生成报告时崩溃

**原因**: 代码不支持 `--model all` 语法

**解决方案**: 使用 `--all` 或 `--model all` 两种方式都可以正常工作
```bash
python benchmark.py --config config.yaml --all
python benchmark.py --config config.yaml --model all
```
