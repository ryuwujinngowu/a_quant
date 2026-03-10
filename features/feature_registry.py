"""
Feature Registry

自动管理所有特征类
"""

from typing import List


class FeatureRegistry:

    _features = []

    @classmethod
    def register(cls, feature):

        cls._features.append(feature)

    @classmethod
    def get_features(cls):

        return cls._features