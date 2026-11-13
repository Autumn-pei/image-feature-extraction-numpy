"""NumPy 数值计算练习（NP-E1~E8 基础，NP-M1~M4 中等）。

运行：python numpy_homework.py
约定：开头 import numpy as np；函数题保留题目指定函数名；
浮点保留完整精度，展示保留 6 位；核对误差绝对容忍 1e-6。
"""
import numpy as np

np.set_printoptions(precision=6, floatmode="fixed", suppress=True)


# %% NP-E1 数组、形状与切片 [基础]
def exercise_e1():
    A = np.array([[0.20, -0.50, 1.10, 0.40],
                  [0.80, 0.30, -0.70, 0.60],
                  [-0.40, 0.90, 0.10, -0.20]])
    # 1. ndim / shape / size
    print("E1 A: ndim =", A.ndim, "shape =", A.shape, "size =", A.size)

    # 2. 第 2 行（下标 1）与最后 2 列
    row1 = A[1]
    last2 = A[:, 2:4]
    print("E1 第2行:", row1, "shape =", row1.shape)
    print("E1 最后2列:\n", last2, "shape =", last2.shape)

    # 3. 复制为 B，把 B 的负数改成 0，保持 A 不变
    B = A.copy()
    B[B < 0] = 0
    print("E1 B(负数置0):\n", B)
    print("E1 A 未变:", np.array_equal(A,
          np.array([[0.20, -0.50, 1.10, 0.40],
                    [0.80, 0.30, -0.70, 0.60],
                    [-0.40, 0.90, 0.10, -0.20]])))


# %% NP-E2 逐元素乘法、点乘与长度 [基础]
def exercise_e2():
    a = np.array([0.80, -0.60, 0.70])
    b = np.array([-0.20, 0.90, 0.40])

    # 1. 逐元素乘 a*b 与点乘 a@b
    elem = a * b
    dot = a @ b
    print("E2 a*b:", elem, "shape =", elem.shape)
    print("E2 a@b:", dot, "shape =", dot.shape)

    # 2. 欧氏长度：平方、求和、开平方；再用 np.linalg.norm 核对
    def norm_manual(v):
        return np.sqrt((v * v).sum())

    len_a, len_b = norm_manual(a), norm_manual(b)
    print("E2 ||a|| =", len_a, " ||b|| =", len_b)
    np.testing.assert_allclose(len_a, np.linalg.norm(a), atol=1e-6)
    np.testing.assert_allclose(len_b, np.linalg.norm(b), atol=1e-6)


# %% NP-E3 两个向量的夹角 [基础]
def angle_deg(a, b):
    """返回两个同维非零一维向量的夹角，单位为度。零向量抛 ValueError。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("angle_deg: 输入不能是零向量")
    cos_theta = (a @ b) / (na * nb)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def exercise_e3():
    a = np.array([1.00, -1.00])
    b = np.array([1.00, 0.00])
    deg = angle_deg(a, b)
    print("E3 夹角(度):", deg)
    np.testing.assert_allclose(deg, 45.0, atol=1e-6)


# %% NP-E4 两个点的距离 [基础]
def exercise_e4():
    p = np.array([-0.50, 1.20, 2.00])
    q = np.array([0.80, -0.40, 1.10])
    diff = p - q
    euclidean = np.sqrt((diff * diff).sum())       # 各坐标差平方和再开方
    manhattan = np.abs(diff).sum()                 # 各坐标差绝对值之和
    print("E4 差向量 p-q:", diff)
    print("E4 欧氏距离:", euclidean, " 曼哈顿距离:", manhattan)
    np.testing.assert_allclose(euclidean, np.sqrt(5.06), atol=1e-6)
    np.testing.assert_allclose(manhattan, 3.80, atol=1e-6)


# %% NP-E5 把向量变为单位长度 [基础]
def unit_vector(a):
    """返回方向相同、长度为 1 的新数组；零向量抛 ValueError。不修改原数组。"""
    a = np.asarray(a, dtype=np.float64)
    length = np.linalg.norm(a)
    if length == 0:
        raise ValueError("unit_vector: 零向量无法单位化")
    return a / length


def exercise_e5():
    a = np.array([-1.20, 0.40, 0.80])
    u = unit_vector(a)
    print("E5 单位向量:", u)
    np.testing.assert_allclose(np.linalg.norm(u), 1.0, atol=1e-6)
    print("E5 原数组未改:", np.array_equal(a, np.array([-1.20, 0.40, 0.80])))


# %% NP-E6 沿不同 axis 汇总 [基础]
def exercise_e6():
    A = np.array([[0.20, -0.50, 1.10],
                  [0.80, 0.40, -0.30]])
    col_sum = A.sum(axis=0)   # 沿行方向压缩，每列求和 -> shape (3,)
    row_mean = A.mean(axis=1)  # 沿列方向压缩，每行求均值 -> shape (2,)
    print("E6 sum(axis=0):", col_sum, "shape =", col_sum.shape)
    print("E6 mean(axis=1):", row_mean, "shape =", row_mean.shape)
    print("E6 最大值:", A.max(), "位置(行,列):", np.unravel_index(A.argmax(), A.shape))
    np.testing.assert_allclose(col_sum, [1.00, -0.10, 0.80], atol=1e-6)


# %% NP-E7 一次广播，处理所有行 [基础]
def exercise_e7():
    X = np.array([[0.80, -0.60, 0.70],
                  [-0.40, 0.50, 0.20]])
    scale = np.array([1.50, 0.50, -1.00])
    bias = np.array([-0.10, 0.20, 0.30])
    Y = X * scale + bias  # scale/bias 形状 (3,)，与 X 的每行对齐广播
    print("E7 Y:\n", Y, "shape =", Y.shape)
    np.testing.assert_allclose(Y, [[1.10, -0.10, -0.40], [-0.70, 0.45, 0.10]], atol=1e-6)


# %% NP-E8 把两张图像组成一个 batch [基础]
def exercise_e8():
    a = np.array([[[0.20, -0.40], [0.70, 0.10]],
                  [[0.30, 0.80], [-0.20, 0.50]],
                  [[-0.10, 0.60], [0.40, -0.30]]])  # [C,H,W]
    b = a + 0.10

    # 1. np.stack 在第 0 轴堆叠，新增轴表示图像编号
    batch = np.stack([a, b], axis=0)  # [B,C,H,W]
    print("E8 batch shape:", batch.shape)

    # 2. 提取两张图的蓝色通道（C=2）
    blue = batch[:, 2, :, :]
    print("E8 蓝色通道 shape:", blue.shape)

    # 3. 转为 [B,H,W,C] 顺序，只输出变换后的 shape
    bhwc = batch.transpose(0, 2, 3, 1)
    print("E8 BHWC shape:", bhwc.shape)
    assert batch.shape == (2, 3, 2, 2) and bhwc.shape == (2, 2, 2, 3)


# %% NP-M1 批量计算夹角 [中等]
def angles_to(query, X):
    """计算 query 与 X 每一行的夹角(度)，返回 shape (X.shape[0],)。

    用矩阵乘法完成，不遍历各行；X 中零向量行返回 np.nan；
    query 为零向量抛 ValueError。
    """
    query = np.asarray(query, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if np.linalg.norm(query) == 0:
        raise ValueError("angles_to: query 不能是零向量")

    q_norm = np.linalg.norm(query)
    x_norm = np.linalg.norm(X, axis=1)
    valid = x_norm > 0

    denom = np.where(valid, x_norm * q_norm, 1.0)  # 避免零行除零
    cos = (X @ query) / denom
    cos = np.clip(cos, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos))
    angles[~valid] = np.nan  # 零向量行不参与，返回 nan
    return angles


def exercise_m1():
    query = np.array([0.60, -0.80, 0.00])
    X = np.array([[0.30, -0.40, 0.00],
                  [-0.50, 0.80, 0.00],
                  [0.80, 0.60, 0.00],
                  [0.00, 0.00, 0.00]])
    ang = angles_to(query, X)
    print("M1 夹角(度):", ang)
    np.testing.assert_allclose(ang[:3], [0.0, 175.1355, 90.0], atol=1e-3)
    assert np.isnan(ang[3])


# %% NP-M2 用余弦相似度做 Top-k 检索 [中等]
def top_k_cosine(query, gallery, k):
    """返回 (选中行索引, 对应余弦相似度)，按相似度从高到低。

    相似度 = 点乘 /(两边长度)；零向量行不参与排序；query 为零向量抛 ValueError；
    k 必须为正整数且不超过有效行数，否则 ValueError；同分保持原顺序(stable)。
    """
    query = np.asarray(query, dtype=np.float64)
    gallery = np.asarray(gallery, dtype=np.float64)
    n = gallery.shape[0]
    valid = np.linalg.norm(gallery, axis=1) > 0
    n_valid = int(valid.sum())

    if not isinstance(k, (int, np.integer)) or k <= 0 or k > n_valid:
        raise ValueError("top_k_cosine: k 必须为正整数且不超过有效行数")
    if np.linalg.norm(query) == 0:
        raise ValueError("top_k_cosine: query 不能是零向量")

    q_norm = np.linalg.norm(query)
    g_norm = np.linalg.norm(gallery, axis=1)
    denom = np.where(valid, g_norm * q_norm, 1.0)
    sim = (gallery @ query) / denom
    sim = np.where(valid, sim, -np.inf)  # 零向量排到最后，不参与 Top-k

    order = np.argsort(-sim, kind="stable")  # 降序，同分保持原顺序
    order = order[valid[order]]
    top = order[:k]
    return top, sim[top]


def exercise_m2():
    query = np.array([0.60, -0.80, 0.00])
    gallery = np.array([[0.60, -0.80, 0.00],
                         [0.30, -0.40, 0.00],
                         [0.00, 0.60, 0.00],
                         [-0.60, 0.80, 0.00],
                         [0.00, 0.00, 0.00]])
    ids = np.array(["img_a", "img_b", "img_c", "img_d", "img_zero"])
    idx, sim = top_k_cosine(query, gallery, k=3)
    print("M2 Top-k 索引:", idx, "相似度:", sim)
    print("M2 选中 ID:", ids[idx])
    np.testing.assert_allclose(sim, [1.0, 1.0, -0.8], atol=1e-6)
    assert list(idx) == [0, 1, 2]


# %% NP-M3 所有点对之间的距离矩阵 [中等]
def pairwise_distances(A, B):
    """用广播构造距离矩阵 D，D[i,j] 为 A[i] 到 B[j] 的欧氏距离。

    返回 (D, nearest)：nearest[i] 为 A[i] 距离最近的 B 行索引。
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    # A[:,None,:] -> [Na,1,D]，B[None,:,:] -> [1,Nb,D]，广播相减 -> [Na,Nb,D]
    diff = A[:, None, :] - B[None, :, :]
    D = np.sqrt((diff * diff).sum(axis=-1))
    nearest = D.argmin(axis=1)
    return D, nearest


def exercise_m3():
    A = np.array([[0.20, -0.40],
                  [0.80, 0.30],
                  [-0.50, 0.70]])
    B = np.array([[0.10, 0.20],
                  [-0.60, -0.20]])
    D, nearest = pairwise_distances(A, B)
    print("M3 距离矩阵 D:\n", D, "shape =", D.shape)
    print("M3 每行最近 B 的索引:", nearest)
    assert D.shape == (3, 2) and list(nearest) == [0, 0, 0]


# %% NP-M4 用矩阵运算完成一次前向计算 [中等]
def forward(X, W1, b1, W2, b2):
    """线性 -> ReLU(maximum(Z,0)) -> 线性。W1/W2 按 [输出特征数,输入特征数] 存储。"""
    Z = X @ W1.T + b1
    H = np.maximum(Z, 0.0)
    Y = H @ W2.T + b2
    return Z, H, Y


def exercise_m4():
    X = np.array([[0.80, -0.60, 0.70],
                  [-0.40, -0.50, 0.90]])
    W1 = np.array([[-0.10, 0.20, -0.80],
                   [0.30, -0.50, 0.10]])
    b1 = np.array([-0.15, 0.05])
    W2 = np.array([[0.20, -0.60],
                   [-0.80, 0.52]])
    b2 = np.array([0.10, -0.20])
    Z, H, Y = forward(X, W1, b1, W2, b2)
    print("M4 Z:\n", Z, "\nM4 H:\n", H, "\nM4 Y:\n", Y)
    np.testing.assert_allclose(Z, [[-0.91, 0.66], [-0.93, 0.27]], atol=1e-6)
    np.testing.assert_allclose(H, [[0.0, 0.66], [0.0, 0.27]], atol=1e-6)
    np.testing.assert_allclose(Y, [[-0.296, 0.1432], [-0.062, -0.0596]], atol=1e-6)


if __name__ == "__main__":
    exercise_e1(); print()
    exercise_e2(); print()
    exercise_e3(); print()
    exercise_e4(); print()
    exercise_e5(); print()
    exercise_e6(); print()
    exercise_e7(); print()
    exercise_e8(); print()
    exercise_m1(); print()
    exercise_m2(); print()
    exercise_m3(); print()
    exercise_m4()
    print("\nAll NumPy exercises passed.")
