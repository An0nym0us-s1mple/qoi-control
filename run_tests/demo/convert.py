import numpy as np

vx = np.fromfile("./dataset/Hurricane/Uf48.bin.f32", dtype=np.float32)
vy = np.fromfile("./dataset/Hurricane/Vf48.bin.f32", dtype=np.float32)
vz = np.fromfile("./dataset/Hurricane/Wf48.bin.f32", dtype=np.float32)
vx.astype(np.float64).tofile("./dataset/Hurricane/data/VelocityX.dat")
vy.astype(np.float64).tofile("./dataset/Hurricane/data/VelocityY.dat")
vz.astype(np.float64).tofile("./dataset/Hurricane/data/VelocityZ.dat")


vx = np.fromfile("./dataset/NYX/velocity_x.f32", dtype=np.float32)
vy = np.fromfile("./dataset/NYX/velocity_y.f32", dtype=np.float32)
vz = np.fromfile("./dataset/NYX/velocity_z.f32", dtype=np.float32)
vx.astype(np.float64).tofile("./dataset/NYX/data/VelocityX.dat")
vy.astype(np.float64).tofile("./dataset/NYX/data/VelocityY.dat")
vz.astype(np.float64).tofile("./dataset/NYX/data/VelocityZ.dat")
