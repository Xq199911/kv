# 共享配置文件 - 所有实验脚本共享此配置
# 修改此文件即可统一调整所有实验的参数

# ================= 模型配置 =================
$MODEL_NAME = "Qwen"
$MODEL_PATH = ".\models\Qwen2.5-3B-Instruct"
$PARAMS = ".\StreamingLLM_GPE\configs\params_qwen_inference.json"

# ================= 实验基础配置 =================
$WAIT_K = 15
$MAX_NEW_TOKENS = 4096
$TOTAL_BUDGET = 2048
$MAX_MEMORY_GB = 20.0

# ================= 样本配置 =================
$MAX_SAMPLES = 100  # 完整实验使用100样本
$MIN_SOURCE_LENGTH = 20

# ================= 序列长度配置 =================
$LONG_SEQUENCE_LENGTHS = @(2000, 5000, 10000, 20000)
$BUDGETS = @(2048, 4096, 8192)

# ================= Baseline参数配置 =================
$STREAMINGLLM_WINDOW = 512
$SINK_TOKENS = 128

# ================= 输出目录配置 =================
$BASE_OUTPUT_DIR = ".\output_logs\a_level_paper"

# ================= HAQ-KV配置（如果使用） =================
$RETRIEVAL_BITS = 4
$INDUCTION_BITS = 16
$LOCAL_BITS = 16

