# CVPR2022

# To Do List:
	目前通过网络输出得到的mask只有右手，下一步就是将双手的加进去；
	得到手和物的mask以后，所有参数相当于都有了。然后分别计算各个指标。

# 数据集筛选：
	palm_print:是手掌数据，没有对应的图片，故删除;
	hands:单纯双手动作，没有手物交互过程，故删除

# mask各通道顺序：
	R—— object
	G—— left_hand
	B—— right_hand

# 手的mask生成过程：
	先通过load_contactpose.py生成gt和随机化后的参数。
	再通过run_cotactopt.py对随机的参数进行优化，得到基本和gt对齐的输出结果：out_ho
	最后通过create_mask_from_net.py得到mask结果

# 随手记：
	在contactpose中，网络只估计了最初的手的pose，beta并没有估计。然后把估计的手按照给定的变换矩阵依次变换到图片上。
	在整个过程中，物体的顶点还有faces没有发生任何变化。