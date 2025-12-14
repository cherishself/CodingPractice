import vtk
import numpy as np
import cv2
from PIL import Image

# 读取异常图
img = Image.open(r"path").convert("L")
width, height = img.size
gray_np = np.array(img)

# 创建 VTK 图像数据
image_data = vtk.vtkImageData()
image_data.SetDimensions(width, height, 1)
image_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

for y in range(height):
    for x in range(width):
        image_data.SetScalarComponentFromFloat(x, y, 0, 0, gray_np[y, x])

# 提取几何结构
geometry_filter = vtk.vtkImageDataGeometryFilter()
geometry_filter.SetInputData(image_data)

# 使用灰度值作为高度
warp = vtk.vtkWarpScalar()
warp.SetInputConnection(geometry_filter.GetOutputPort())
warp.SetScaleFactor(0.5)

# 构建与 OpenCV COLORMAP_JET 一致的 LookupTable
lut = vtk.vtkLookupTable()
lut.SetNumberOfTableValues(256)
lut.Build()

jet_colors = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
jet_colors = jet_colors.reshape(-1, 3)

for i in range(256):
    r, g, b = jet_colors[i] / 255.0
    lut.SetTableValue(i, r, g, b, 1.0)

# 映射颜色
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(warp.GetOutputPort())
mapper.SetLookupTable(lut)
mapper.SetScalarRange(0, 255)
mapper.SetScalarVisibility(True)   # 打开标量可见性

actor = vtk.vtkActor()
actor.SetMapper(mapper)

# 只保留曲面表面
actor.GetProperty().SetRepresentationToSurface()
actor.GetProperty().EdgeVisibilityOff()
actor.GetProperty().SetInterpolationToPhong()

# 渲染设置
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(1, 1, 1)

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

render_window.Render()
interactor.Start()
