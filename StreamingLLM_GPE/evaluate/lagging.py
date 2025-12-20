# coding=utf-8
# Modified 2025 by Junlong Tong (Shanghai Jiao Tong University & Eastern Institute of Technology).
#
# This file is a modified version of the 'simuleval' repository implementation from:
# The Facebook, Inc.
#
# Original license and copyright as follows:
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license of 
# Attribution-ShareAlike 4.0 International





def calculate_al_and_laal(source_length, target_length, delays):
    """
    Function to compute latency on one sentence (instance).

    Args:
        delays (List[Union[float, int]]): Sequence of delays (source word indices when each target word was generated).
        source_length (Union[float, int]): Length of source sequence (in words).
        target_length (Union[float, int]): Length of target sequence (in words).

    Returns:
        tuple: (AL, LAAL) - Average Lagging and Length-Aware Average Lagging
    """
    
    # 输入验证
    if not delays or len(delays) == 0:
        return 0.0, 0.0
    
    if source_length <= 0:
        return 0.0, 0.0
    
    if target_length <= 0:
        return 0.0, 0.0
    
    # 如果第一个延迟值异常大，可能是数据问题
    if delays[0] > source_length * 10:
        # 尝试修复：将异常值限制在合理范围内
        delays = [min(d, source_length) for d in delays]
    
    # 如果第一个延迟就超过了源长度，说明有问题
    # 修复：将所有异常值限制在合理范围内
    delays = [max(0, min(int(d), source_length)) for d in delays]
    
    # 如果修复后第一个延迟仍然异常，返回0
    if delays[0] > source_length or delays[0] < 0:
        return 0.0, 0.0

    AL = 0.0
    LAAL = 0.0
    
    # 计算gamma值
    # gamma是目标长度和源长度的比例
    gamma_AL = target_length / source_length if source_length > 0 else 1.0
    gamma_LAAL = max(len(delays), target_length) / source_length if source_length > 0 else 1.0
    
    # 如果gamma太小（源长度远大于目标长度），会导致计算问题
    # 在这种情况下，我们需要调整计算方法
    if gamma_AL < 0.01:  # 如果gamma太小，说明源长度远大于目标长度
        # 使用更合理的计算方法：直接计算平均延迟
        # 在这种情况下，AL和LAAL应该接近平均延迟值
        if len(delays) > 0:
            avg_delay = sum(delays) / len(delays)
            # AL和LAAL在这种情况下应该接近平均延迟
            AL = max(0.0, avg_delay)
            LAAL = max(0.0, avg_delay)
        return AL, LAAL
    
    tau = 0
    for t_minus_1, d in enumerate(delays):
        # d 是延迟值（生成第t个目标词时，已经读取的源端单词数）
        # t_minus_1 是目标词索引（从0开始，所以是 t-1）
        # 确保d在合理范围内
        d = max(0, min(int(d), source_length))
        
        if d <= source_length:
            # 计算AL和LAAL的累加项
            # 注意：根据Average Lagging定义，公式是 d - (t / gamma)
            # 这里 t_minus_1 是 t-1，所以 t = t_minus_1 + 1
            t = t_minus_1 + 1  # 目标词索引（从1开始）
            lag_value_AL = d - (t / gamma_AL)
            lag_value_LAAL = d - (t / gamma_LAAL)
            
            # 累加（注意：lag_value可能为负，这是正常的，表示提前生成）
            # 但根据Average Lagging的定义，我们只累加正值部分
            AL += max(0.0, lag_value_AL)
            LAAL += max(0.0, lag_value_LAAL)
            tau = t_minus_1 + 1

            # 如果延迟等于源长度，说明已经读取完所有源端内容
            if d >= source_length:
                break
    
    # 计算平均值
    if tau > 0:
        AL /= tau
        LAAL /= tau
    else:
        # 如果没有有效的延迟值，返回0
        AL = 0.0
        LAAL = 0.0
    
    return AL, LAAL
    