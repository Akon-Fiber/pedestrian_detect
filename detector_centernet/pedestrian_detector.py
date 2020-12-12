# -*- coding: utf-8 -*-


class PedestrianDetector(object):
    """
    行人人头检测基础类，可以衍生出各检测算法对应的子类
    Attributes:
        detect: 对于给定输入进行行人人头检测
    """

    def __init__(self):
        """
        初始化过程
        """
        return

    def detect(self, DetectInput):
        """
        对于给定输入进行行人人头检测
        :param DetectInput: 行人人头检测输入
        """
        raise NotImplementedError
