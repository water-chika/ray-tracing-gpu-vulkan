import ctypes

ray = ctypes.cdll.LoadLibrary('./Debug/ray_trace.dll')

print(dir(ray))

ray.ray_trace(100, 10, False, 1920, 1080)
