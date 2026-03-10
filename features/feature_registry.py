"""
因子注册中心：统一管理所有特征类，支持灵活选择、批量调度
"""
from typing import List, Dict, Type
from features.base_feature import BaseFeature


class FeatureRegistry:
    """因子注册器，单例模式全局唯一"""
    _instance = None
    _registry: Dict[str, Type[BaseFeature]] = {}

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    @classmethod
    def register(cls, feature_name: str):
        """
        因子注册装饰器，新增因子只需加此装饰器即可自动注册
        :param feature_name: 因子唯一标识，不可重复
        """
        def wrapper(feature_class: Type[BaseFeature]):
            if feature_name in cls._registry:
                raise ValueError(f"因子{feature_name}已注册，不可重复注册")
            feature_class.feature_name = feature_name
            cls._registry[feature_name] = feature_class
            return feature_class
        return wrapper

    @classmethod
    def get_feature(cls, feature_name: str) -> BaseFeature:
        """根据因子名称获取特征实例"""
        if feature_name not in cls._registry:
            raise ValueError(f"因子{feature_name}未注册，可用因子：{list(cls._registry.keys())}")
        return cls._registry[feature_name]()

    @classmethod
    def get_features(cls, feature_name_list: List[str]) -> List[BaseFeature]:
        """批量获取因子实例，支持灵活选择因子"""
        return [cls.get_feature(name) for name in feature_name_list]

    @classmethod
    def get_all_features(cls) -> List[BaseFeature]:
        """获取所有已注册的因子实例"""
        return [cls() for cls in cls._registry.values()]

    @classmethod
    def list_all_features(cls) -> List[str]:
        """列出所有已注册的因子名称"""
        return list(cls._registry.keys())


# 全局单例实例
feature_registry = FeatureRegistry()