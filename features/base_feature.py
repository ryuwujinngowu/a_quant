# features/base_feature.py
# 导入pandas：处理量化数据的核心库，你的K线/行情数据都存在DataFrame里
import pandas as pd
# 导入ABC（抽象基类）和abstractmethod（抽象方法装饰器）：
# 作用是给所有特征定"规矩"，强制子类必须按规则写代码
from abc import ABC, abstractmethod


# 定义特征基类，继承ABC（抽象基类）：
# 这个类不能直接用，只能被其他特征类（比如涨停数、均线）继承
class BaseFeature(ABC):
    # 初始化方法：创建特征实例时自动执行
    def __init__(self, data_api=None):
        # data_api：预留的参数，以后你可以传数据接口（比如从数据库/接口拿数据）
        self.data_api = data_api
        # feature_name：自动把特征类的名字作为特征名（比如MAFeature的名字就是"MAFeature"）
        # 方便后续机器学习识别不同特征
        self.feature_name = self.__class__.__name__
        # _locked：参数锁定开关，默认打开（True）
        # 作用：防止你中途改特征的计算参数，保证特征口径统一
        self._locked = True

    # @abstractmethod：抽象方法装饰器，这是"强制规矩"
    # 所有继承BaseFeature的子类（比如涨停数特征），必须实现这个calculate方法
    # 不实现的话，代码运行时直接报错，避免你漏写核心逻辑
    # 入参df：必须是包含股票数据的DataFrame；返回值也必须是DataFrame
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        pass  # 空方法，留给子类写具体的特征计算逻辑（比如算涨停数、均线）

    # run方法：所有特征的"统一执行入口"
    # 你外面调用特征时，只需要调run()，不用管内部细节
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        # 第一步：强制安全检查——必须有trade_date列（交易日）
        # 没有的话直接报错，避免你传错数据导致计算出错
        if "trade_date" not in df.columns:
            raise ValueError("特征计算必须包含 trade_date 列")

        # 第二步：强制按股票代码（ts_code）+ 交易日（trade_date）排序
        # 核心作用：从根源防止"未来函数"（比如用明天的数据算今天的特征）
        # ignore_index=True：重置索引，避免排序后索引乱了
        df = df.sort_values(["ts_code", "trade_date"], ignore_index=True)

        # 第三步：调用子类实现的calculate方法，计算具体特征
        # 比如涨停数特征的calculate会算limit_up_count列
        return self.calculate(df)

    # __setattr__：重写Python的属性赋值逻辑（比如self.ma_periods=5这种赋值）
    # 核心作用：锁定已定义的参数，防止你中途改参数导致特征口径不一致
    def __setattr__(self, key, value):
        # 条件1：_locked开关已打开（True）
        # 条件2：要改的属性不是_locked本身
        if hasattr(self, "_locked") and self._locked and key != "_locked":
            # 条件3：要改的属性已经存在（比如你之前定义了self.ma_periods=5，现在想改成10）
            if key in self.__dict__:
                # 直接报错，阻止你改参数——保证特征计算口径永远不变
                raise RuntimeError(f"参数 {key} 已锁定，保证特征口径统一")
        # 如果不触发上面的条件，就按正常逻辑赋值（比如初始化时赋值）
        super().__setattr__(key, value)