# ===================== 回测基础配置 =====================
DEFAULT_INIT_CAPITAL = 10000  # 默认初始本金1W元
COMMISSION_RATE = 0.0001       # 交易手续费（万1，买卖双向收取）
STAMP_DUTY_RATE = 0.001        # 印花税（千1，仅卖出收取，A股规则）
SLIPPAGE_RATE = 0.001          # 滑点（买卖均按成交价的0.1%计算，模拟真实成交）

# 分仓策略配置
MAX_POSITION_COUNT = 5  # 最大持仓数量，总仓位分5等份
# 涨停策略配置
# DAILY_LIMIT_UP_RATE = 0.098  # 涨停判断阈值（9.8%，避免四舍五入误差）
# ST_LIMIT_UP_RATE = 0.048  # ST股票涨停阈值（后续扩展用）

# 涨停阈值（带容错）
MAIN_BOARD_LIMIT_UP_RATE    = 0.098    # 主板：10% 涨停
STAR_BOARD_LIMIT_UP_RATE    = 0.198    # 创业板/科创板：20% 涨停
# ST_BOARD_LIMIT_UP_RATE      = 0.048    # ST：5% 涨停（暂时不用）

# ===================== A股交易规则 =====================
T_PLUS_1 = True                 # T+1交易规则
MIN_TRADE_VOLUME = 100          # 最低交易单位1手=100股
DAILY_LIMIT_UP_RATE = 0.098    # 涨停判断阈值（9.8%，避免四舍五入误差）
DAILY_LIMIT_DOWN_RATE = 0.098  # 跌停判断阈值

# ===================== 策略默认参数 =====================
DEFAULT_HOLD_DAYS = 3           # 涨停策略默认持有天数
DEFAULT_STOP_LOSS_RATE = 0.05   # 涨停策略默认止损比例5%
RISK_FREE_RATE = 0.03           # 无风险年化收益率（用于夏普比率计算）
ANNUAL_TRADE_DAYS = 252         # 年化交易日天数