import numpy as np
from numpy.linalg import svd, lstsq

def matrix_pencil_features(segment, M=10, L=None):
    N = len(segment)
    if L is None:
        L = N // 2
    rows = N - L
    cols = L + 1
    Y = np.array([segment[i:i+cols] for i in range(rows)], dtype=np.float64)
    U, s, Vh = svd(Y, full_matrices=False)
    s_norm = s[:M] / (s[0] + 1e-10)
    Vm = Vh[:M, :]
    Y1 = Vm[:, :-1]
    Y2 = Vm[:, 1:]
    Z, _, _, _ = lstsq(Y1.T, Y2.T, rcond=None)
    Z = Z.T
    eigenvalues = np.linalg.eigvals(Z)
    idx = np.argsort(np.abs(eigenvalues))[::-1][:M]
    poles = eigenvalues[idx]
    features = np.concatenate([
        s_norm,
        np.abs(poles),
        np.angle(poles) / np.pi
    ])
    return features.astype(np.float32)

X = np.load('X.npy')
print(f"Пересчитываю признаки с M=10 для {len(X)} сегментов...")
features = []
for i, seg in enumerate(X):
    features.append(matrix_pencil_features(seg, M=10))
    if (i+1) % 10000 == 0:
        print(f"  {i+1}/{len(X)}")

F = np.array(features, dtype=np.float32)
np.save('F.npy', F)
print(f"Готово! Форма: {F.shape}")  # должно быть (109468, 30)